import gc
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import zarr
from omegaconf import DictConfig
from src.models.aacnn import AACNN


def _open_z_arr(zarr_path: str, image_name: str):
    store = zarr.open(zarr_path, mode="r")

    if "images" in store and image_name in store["images"]:
        grp = store[f"images/{image_name}"]

        if "data" in grp:
            # microplastics layout: spatial cube (H, W, n_wn)
            z_arr = grp["data"]
            H, W, Bands = z_arr.shape
            return z_arr, H, W, Bands

        elif "X" in grp:
            # PCUK layout: flat annotated spectra (N, n_wn)
            X     = grp["X"][:]
            z_arr = X[:, np.newaxis, :]   # (N, 1, n_wn)
            H, W, Bands = z_arr.shape
            return z_arr, H, W, Bands

        else:
            raise KeyError(
                f"'{image_name}' found in store but has neither 'data' nor 'X'.\n"
                f"  Keys: {list(grp.keys())}"
            )

    elif image_name in store:
        # legacy test-store layout: <name>/X at root level
        X     = store[f"{image_name}/X"][:]
        z_arr = X[:, np.newaxis, :]
        H, W, Bands = z_arr.shape
        return z_arr, H, W, Bands

    else:
        raise KeyError(
            f"'{image_name}' not found in '{zarr_path}'.\n"
            f"  Top-level keys: {list(store.keys())}"
        )



def inference_worker(
    gpu_id: int,
    ckpt_path: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    zarr_path: str,
    image_name: str,
) -> None:
    try:
        device = torch.device(f"cuda:{gpu_id}")
        model = AACNN.load_from_checkpoint(ckpt_path, weights_only=False).to(device).half()
        model.eval()

        z_arr, H, W, Bands = _open_z_arr(zarr_path, image_name)  # ← shared helper
        flat_data_ram = z_arr[:].reshape(-1, Bands)
        del z_arr
        gc.collect()

        with torch.inference_mode():
            while True:
                msg = input_queue.get()
                if msg is None:
                    break
                start, end = msg
                data = flat_data_ram[start:end]
                chunk = (
                    torch.from_numpy(data.astype(np.float32))
                    .unsqueeze(1)
                    .to(device, non_blocking=True)
                    .half()
                )
                logits = model(chunk)
                probs = F.softmax(logits, dim=1)
                output_queue.put((start, probs.cpu().numpy()))
                del data, chunk

    except Exception as e:
        import traceback
        print(f"[worker gpu={gpu_id}] CRASHED: {e}", flush=True)
        traceback.print_exc()
        while True:
            msg = input_queue.get()
            if msg is None:
                break
        output_queue.put((-1, None))


def _run_inference_single_gpu(
    gpu_id: int,
    ckpt_path: str,
    zarr_path: str,
    image_name: str,
    batch_size: int,
) -> np.ndarray:
    """Direct, no-multiprocessing inference for one image on one GPU.

    Skips the process spawn, CUDA-context init, and queue IPC that
    _run_inference_multi_gpu pays even when there's only one device to use.
    For this codebase's small AACNN model those fixed per-image costs
    typically dwarf the actual forward-pass time, so this path is not just
    simpler but usually faster whenever only one GPU is configured. Falls
    back to CPU if CUDA isn't available (e.g. local/dev machines).
    """
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    model = AACNN.load_from_checkpoint(ckpt_path, weights_only=False).to(device).half()
    model.eval()

    z_arr, H, W, Bands = _open_z_arr(zarr_path, image_name)
    flat_data  = z_arr[:].reshape(-1, Bands)
    num_pixels = H * W
    prob_map   = None

    with torch.inference_mode():
        for start in range(0, num_pixels, batch_size):
            end   = min(start + batch_size, num_pixels)
            chunk = (
                torch.from_numpy(flat_data[start:end].astype(np.float32))
                .unsqueeze(1)
                .to(device, non_blocking=True)
                .half()
            )
            probs = F.softmax(model(chunk), dim=1).cpu().numpy()
            if prob_map is None:
                n_classes = probs.shape[1]
                prob_map  = np.zeros((num_pixels, n_classes), dtype=np.float16)
            prob_map[start:end] = probs

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return prob_map.reshape(H, W, n_classes)


def _run_inference_multi_gpu(
    devices: list[int],
    ckpt_path: str,
    zarr_path: str,
    image_name: str,
    batch_size: int,
) -> np.ndarray:
    """Split one image's pixels across len(devices) GPU worker processes."""
    _, H, W, Bands = _open_z_arr(zarr_path, image_name)  # shape only, workers load data
    num_pixels = H * W

    input_queue  = mp.Queue()
    output_queue = mp.Queue()
    processes    = []

    for gpu_id in devices:
        p = mp.Process(
            target=inference_worker,
            args=(gpu_id, ckpt_path, input_queue, output_queue, zarr_path, image_name),
        )
        p.start()
        processes.append(p)

    for i in range(0, num_pixels, batch_size):
        input_queue.put((i, min(i + batch_size, num_pixels)))
    for _ in range(len(processes)):
        input_queue.put(None)

    num_batches = (num_pixels + batch_size - 1) // batch_size
    prob_map    = None

    for _ in range(num_batches):
        start_idx, probs = output_queue.get()
        if start_idx == -1:
            raise RuntimeError(f"inference_worker crashed for {image_name}")
        if prob_map is None:
            n_classes = probs.shape[1]
            prob_map  = np.zeros((num_pixels, n_classes), dtype=np.float16)
        prob_map[start_idx: start_idx + len(probs)] = probs

    for p in processes:
        p.join()

    return prob_map.reshape(H, W, n_classes)


def run_inference(
    cfg: DictConfig | None = None,
    image_name: str = "",
    ckpt_path: str = "",
    batch_size: int = 512,
    zarr_path: str | None = None,
    devices: list[int] | None = None,
) -> np.ndarray:
    """Run pixel-wise inference on one image.

    Dispatches at runtime purely on how many entries are in `devices`
    (from cfg.inference.devices, or passed explicitly) — no separate flag
    needed, since the device list is already the single source of truth
    for "how many GPUs, and which ones":
      - 0 or 1 devices → _run_inference_single_gpu: loads the model and
        runs directly in this process, no multiprocessing.
      - 2+ devices     → _run_inference_multi_gpu: the mp.Process worker
        pool, splitting this image's pixels across all listed GPUs.

    Can be called with a full Hydra cfg (as in main.py) or standalone with
    explicit arguments — cfg is not required when zarr_path and devices are
    provided directly.

    Args:
        cfg:        Hydra DictConfig (optional — used by main.py).
        image_name: Key of the image inside the zarr store.
        ckpt_path:  Path to the Lightning checkpoint.
        batch_size: Number of pixels per GPU batch.
        zarr_path:  Path to the zarr store. Falls back to cfg.data.zarr_path
                    when cfg is provided and zarr_path is None.
        devices:    List of GPU ids, e.g. [0] or [0, 1]. Defaults to [0] when
                    cfg is not provided.
    """
    if cfg is not None:
        zarr_path = zarr_path or cfg.data.zarr_path
        devices   = list(cfg.inference.devices)
    else:
        if not zarr_path:
            raise ValueError("zarr_path is required when cfg is not provided")
        devices = list(devices) if devices is not None else [0]

    if len(devices) <= 1:
        gpu_id = devices[0] if devices else 0
        return _run_inference_single_gpu(gpu_id, ckpt_path, zarr_path, image_name, batch_size)

    return _run_inference_multi_gpu(devices, ckpt_path, zarr_path, image_name, batch_size)


def predict_array(
    spectra: np.ndarray,
    ckpt_path: str,
    batch_size: int = 512,
    device: int | str = 0,
    z_normalize: bool = False,
) -> np.ndarray:
    """Run inference directly on a numpy array — no zarr store required.

    This is the entry point for notebooks, scripts, and non-zarr pipelines.
    It runs on a single device (no multiprocessing), so it is straightforward
    to call interactively.

    Args:
        spectra:     Hyperspectral data as a numpy array.
                     Shape ``(H, W, L)`` for a 2-D image, or ``(N, L)`` for a
                     flat list of spectra.  dtype can be float32 or float64;
                     conversion is handled internally.
        ckpt_path:   Path to the Lightning ``.ckpt`` checkpoint file.
        batch_size:  Number of spectra processed per forward pass.
                     Reduce if you run out of GPU memory.
        device:      GPU index (``0``, ``1``, …) or ``"cpu"`` for CPU-only
                     inference.  Falls back to CPU automatically when CUDA is
                     not available.
        z_normalize: Apply per-spectrum z-score normalisation before inference.
                     Set this to ``True`` if ``data.z_normalize: True`` was
                     used during training and the model does not include an
                     internal normalisation layer.

    Returns:
        Probability maps as float16:

        - Input ``(H, W, L)`` → output ``(H, W, n_classes)``
        - Input ``(N, L)``    → output ``(N, n_classes)``

    Example::

        import numpy as np
        from src.inference.inference_engine import predict_array

        cube = np.load("my_image.npy")          # (H, W, L)
        prob_map = predict_array(cube, "checkpoints/best.ckpt")
        argmax   = prob_map.argmax(-1)           # (H, W)
    """
    if spectra.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2-D (N, L) or 3-D (H, W, L) array; got shape {spectra.shape}"
        )

    # ── flatten to (N, L) ─────────────────────────────────────────────────────
    is_image = spectra.ndim == 3
    if is_image:
        H, W, L = spectra.shape
        flat = spectra.reshape(-1, L).astype(np.float32)
    else:
        flat = spectra.astype(np.float32)
        H = W = None

    N = len(flat)

    # ── optional z-normalisation ──────────────────────────────────────────────
    if z_normalize:
        mu    = flat.mean(axis=1, keepdims=True)
        sigma = flat.std(axis=1, keepdims=True)
        flat  = (flat - mu) / (sigma + 1e-8)

    # ── resolve device ────────────────────────────────────────────────────────
    if isinstance(device, int):
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{device}")
        else:
            print(
                "Warning: CUDA not available — running on CPU (may be slow).",
                file=sys.stderr,
            )
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # ── load model ────────────────────────────────────────────────────────────
    model = (
        AACNN.load_from_checkpoint(ckpt_path, weights_only=False)
        .to(dev)
        .half()
        .eval()
    )

    # ── batched forward pass ──────────────────────────────────────────────────
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            chunk = (
                torch.from_numpy(flat[start:end])
                .unsqueeze(1)           # (B, 1, L)
                .to(dev, non_blocking=True)
                .half()
            )
            probs = F.softmax(model(chunk), dim=1).cpu().numpy().astype(np.float16)
            results.append(probs)

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ── reassemble ────────────────────────────────────────────────────────────
    prob_flat = np.concatenate(results, axis=0)   # (N, n_classes)
    n_classes = prob_flat.shape[1]

    if is_image:
        return prob_flat.reshape(H, W, n_classes)
    return prob_flat

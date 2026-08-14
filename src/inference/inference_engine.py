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
    """Return a *lazy* handle to an image's spectra plus its shape.

    Importantly this never materialises the array — the caller reads only
    the pixel ranges it actually needs via `_read_pixel_range`, so opening
    an image costs ~nothing and holding many open handles (e.g. one per GPU
    worker) doesn't multiply memory use.
    """
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
            z_arr = grp["X"]
            N, Bands = z_arr.shape
            return z_arr, N, 1, Bands

        else:
            raise KeyError(
                f"'{image_name}' found in store but has neither 'data' nor 'X'.\n"
                f"  Keys: {list(grp.keys())}"
            )

    elif image_name in store:
        # legacy test-store layout: <name>/X at root level
        z_arr = store[f"{image_name}/X"]
        N, Bands = z_arr.shape
        return z_arr, N, 1, Bands

    else:
        raise KeyError(
            f"'{image_name}' not found in '{zarr_path}'.\n"
            f"  Top-level keys: {list(store.keys())}"
        )


def _read_pixel_range(z_arr, W: int, Bands: int, start: int, end: int) -> np.ndarray:
    """Read pixels [start, end) of the flattened (H*W) index range directly
    off the zarr array, without ever loading the whole image into RAM.

    - Flat layout (z_arr.ndim == 2, W == 1): pixel index == row index, so
      it's a straight chunked slice.
    - Spatial cube layout (z_arr.ndim == 3): translate the flat range into
      the whole rows that cover it, slice those (zarr only pulls the
      chunks it needs), then trim to the exact pixel range.
    """
    if z_arr.ndim == 2:
        return np.asarray(z_arr[start:end])

    row_start = start // W
    row_end   = (end - 1) // W + 1
    rows = np.asarray(z_arr[row_start:row_end]).reshape(-1, Bands)
    lo = start - row_start * W
    return rows[lo: lo + (end - start)]


def _load_model(ckpt_path: str, device: torch.device):
    model = AACNN.load_from_checkpoint(ckpt_path, weights_only=False).to(device).half()
    model.eval()
    return model


def _infer_batches(model, device: torch.device, z_arr, H: int, W: int, Bands: int, batch_size: int) -> np.ndarray:
    """Run the model over one image's pixels in batches, reading each batch
    lazily from `z_arr`. Shared by the single-GPU path and each persistent
    worker's per-task handling."""
    num_pixels = H * W
    prob_map   = None

    with torch.inference_mode():
        for start in range(0, num_pixels, batch_size):
            end   = min(start + batch_size, num_pixels)
            data  = _read_pixel_range(z_arr, W, Bands, start, end)
            chunk = (
                torch.from_numpy(data.astype(np.float32))
                .unsqueeze(1)
                .to(device, non_blocking=True)
                .half()
            )
            probs = F.softmax(model(chunk), dim=1).cpu().numpy()
            if prob_map is None:
                n_classes = probs.shape[1]
                prob_map  = np.zeros((num_pixels, n_classes), dtype=np.float16)
            prob_map[start:end] = probs

    return prob_map.reshape(H, W, prob_map.shape[-1])


def _persistent_gpu_worker(
    gpu_id: int,
    ckpt_path: str,
    task_queue: mp.Queue,
    output_queue: mp.Queue,
) -> None:
    """Long-lived worker: loads the checkpoint once, then serves batches for
    however many images are sent its way until it receives the sentinel.

    Replaces the old design where a fresh process (and a fresh checkpoint
    load) was spawned per image. It also never loads a full image into RAM
    — each task carries a pixel range, read lazily via `_read_pixel_range`.

    Tasks and results are tagged with an `epoch` (bumped once per `infer()`
    call by the session). Queues are now shared across the whole run rather
    than recreated per image, so if a worker dies mid-image any message it
    already had in flight — or a straggler from a worker that's simply slow
    — must not be mistaken for a result belonging to the *next* image.
    Tagging lets the receiver (`InferenceSession._infer_multi`) discard
    anything that isn't from the epoch it's currently waiting on.
    """
    epoch = 0
    try:
        device = torch.device(f"cuda:{gpu_id}")
        model  = _load_model(ckpt_path, device)

        cached_key = None
        cached_arr = cached_W = cached_Bands = None

        with torch.inference_mode():
            while True:
                msg = task_queue.get()
                if msg is None:
                    break
                epoch, zarr_path, image_name, start, end = msg

                key = (zarr_path, image_name)
                if key != cached_key:
                    cached_arr, _, cached_W, cached_Bands = _open_z_arr(zarr_path, image_name)
                    cached_key = key

                data = _read_pixel_range(cached_arr, cached_W, cached_Bands, start, end)
                chunk = (
                    torch.from_numpy(data.astype(np.float32))
                    .unsqueeze(1)
                    .to(device, non_blocking=True)
                    .half()
                )
                probs = F.softmax(model(chunk), dim=1).cpu().numpy()
                output_queue.put((epoch, start, probs))

    except Exception as e:
        import traceback
        print(f"[worker gpu={gpu_id}] CRASHED: {e}", flush=True)
        traceback.print_exc()
        output_queue.put((epoch, -1, None))


class InferenceSession:
    """Holds a loaded model (single GPU) or a pool of persistent GPU worker
    processes (multi-GPU) for the lifetime of an inference run.

    Use this instead of calling `run_inference` in a loop — it loads the
    checkpoint exactly once per device instead of once per image, and reads
    input data lazily per batch instead of materialising a full image per
    call. Use as a context manager so workers are always torn down:

        with InferenceSession(ckpt_path, devices=[0, 1], batch_size=512) as sess:
            for img in test_images:
                prob_map = sess.infer(zarr_path, img["name"])
    """

    def __init__(self, ckpt_path: str, devices: list[int] | None = None, batch_size: int = 512):
        self.ckpt_path  = ckpt_path
        self.devices    = list(devices) if devices else [0]
        self.batch_size = batch_size
        self.multi_gpu  = len(self.devices) > 1
        self._closed    = False
        self._epoch     = 0

        if self.multi_gpu:
            self.task_queue   = mp.Queue()
            self.output_queue = mp.Queue()
            self.workers = [
                mp.Process(
                    target=_persistent_gpu_worker,
                    args=(gpu_id, ckpt_path, self.task_queue, self.output_queue),
                )
                for gpu_id in self.devices
            ]
            for w in self.workers:
                w.start()
        else:
            gpu_id = self.devices[0] if self.devices else 0
            self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
            self.model  = _load_model(ckpt_path, self.device)

    def infer(self, zarr_path: str, image_name: str) -> np.ndarray:
        z_arr, H, W, Bands = _open_z_arr(zarr_path, image_name)
        if not self.multi_gpu:
            return _infer_batches(self.model, self.device, z_arr, H, W, Bands, self.batch_size)
        return self._infer_multi(zarr_path, image_name, H, W, Bands)

    def _infer_multi(self, zarr_path: str, image_name: str, H: int, W: int, Bands: int) -> np.ndarray:
        self._epoch += 1
        epoch       = self._epoch
        num_pixels  = H * W
        starts      = list(range(0, num_pixels, self.batch_size))
        for start in starts:
            end = min(start + self.batch_size, num_pixels)
            self.task_queue.put((epoch, zarr_path, image_name, start, end))

        prob_map  = None
        n_classes = None
        received  = 0
        while received < len(starts):
            tag, start, probs = self.output_queue.get()
            if start == -1:
                raise RuntimeError(f"inference worker crashed on {image_name}")
            if tag != epoch:
                # Straggler from a previous (possibly crashed) image's tasks
                # — the queues are shared across the whole run, so discard
                # anything that isn't from the epoch we're waiting on.
                continue
            if prob_map is None:
                n_classes = probs.shape[1]
                prob_map  = np.zeros((num_pixels, n_classes), dtype=np.float16)
            prob_map[start:start + len(probs)] = probs
            received += 1

        return prob_map.reshape(H, W, n_classes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.multi_gpu:
            for _ in self.workers:
                self.task_queue.put(None)
            for w in self.workers:
                w.join()
        else:
            del self.model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_inference(
    cfg: DictConfig | None = None,
    image_name: str = "",
    ckpt_path: str = "",
    batch_size: int = 512,
    zarr_path: str | None = None,
    devices: list[int] | None = None,
) -> np.ndarray:
    """Run pixel-wise inference on one image.

    Convenience one-shot wrapper around `InferenceSession` for callers that
    only need a single image (e.g. `predict.py`). For a full test set,
    open one `InferenceSession` and call `.infer(...)` per image instead —
    this function reloads the checkpoint every call, which is exactly the
    per-image reload cost `InferenceSession` exists to avoid.

    Dispatches at runtime purely on how many entries are in `devices`
    (from cfg.inference.devices, or passed explicitly) — no separate flag
    needed, since the device list is already the single source of truth
    for "how many GPUs, and which ones":
      - 0 or 1 devices → loads the model and runs directly in this
        process, no multiprocessing.
      - 2+ devices     → a short-lived worker pool, splitting this image's
        pixels across all listed GPUs.

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

    with InferenceSession(ckpt_path, devices=devices, batch_size=batch_size) as session:
        return session.infer(zarr_path, image_name)


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

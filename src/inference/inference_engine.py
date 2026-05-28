import gc
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import zarr
from omegaconf import DictConfig
from src.models.aacnn import AACNN


def _open_z_arr(zarr_path: str, image_name: str):
    """
    Open the correct array from either store layout.
    Train store: images/<name>/data  shape (H, W, n_wn)
    Test store:  <name>/X            shape (n, n_wn) — returned as (n, 1, n_wn)
    Returns (z_arr, H, W, Bands) where z_arr is always (H, W, n_wn)-compatible.
    """
    store = zarr.open(zarr_path, mode="r")
    if "images" in store and image_name in store["images"]:
        z_arr = store[f"images/{image_name}/data"]   # lazy zarr array
        H, W, Bands = z_arr.shape
        return z_arr, H, W, Bands
    elif image_name in store:
        X = store[f"{image_name}/X"][:]              # (n, n_wn) — load fully
        z_arr = X[:, np.newaxis, :]                  # (n, 1, n_wn) numpy array
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


def run_inference(
    cfg: DictConfig,
    image_name: str,
    ckpt_path: str,
    batch_size: int = 512,
    zarr_path: str = None,
) -> np.ndarray:
    zarr_path = zarr_path or cfg.data.zarr_path

    _, H, W, Bands = _open_z_arr(zarr_path, image_name)  # shape only, workers load data
    num_pixels = H * W

    input_queue  = mp.Queue()
    output_queue = mp.Queue()
    processes    = []

    for gpu_id in cfg.inference.devices:
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

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from dataclasses import dataclass, field
import gc
import zarr
import torch
import numpy as np
import torch.multiprocessing as mp

from src.Models.aacnn import AACNN


def inference_worker(
    gpu_id, ckpt_path, input_queue, output_queue, zarr_path, image_name
):
    device = torch.device(f"cuda:{gpu_id}")

    model = AACNN.load_from_checkpoint(ckpt_path, weights_only=False).to(device).half()
    model.eval()

    store = zarr.open(zarr_path, mode="r")
    z_arr = store[f"images/{image_name}/data"]
    flat_data_ram = z_arr[:].reshape(-1, z_arr.shape[2])

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


def run_inference(cfg, image_name, ckpt_path, batch_size=512):

    zarr_path = cfg.data.zarr_path
    store = zarr.open(zarr_path, mode="r")
    H, W, Bands = store[f"images/{image_name}/data"].shape
    num_pixels = H * W

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    processes = []
    for gpu_id in [0, 1]:
        p = mp.Process(
            target=inference_worker,
            args=(gpu_id, ckpt_path, input_queue, output_queue, zarr_path, image_name),
        )
        p.start()
        processes.append(p)

    for i in range(0, num_pixels, batch_size):
        end_idx = min(i + batch_size, num_pixels)
        input_queue.put((i, end_idx))

    for _ in range(len(processes)):
        input_queue.put(None)

    num_batches = (num_pixels + batch_size - 1) // batch_size
    prob_map = None  # shape: (num_pixels, n_classes), float16

    for _ in range(num_batches):
        start_idx, probs = output_queue.get()  # probs: (batch, n_classes)

        if prob_map is None:  # first result reveals n_classes
            n_classes = probs.shape[1]
            prob_map = np.zeros((num_pixels, n_classes), dtype=np.float16)

        prob_map[start_idx : start_idx + len(probs)] = probs

    for p in processes:
        p.join()

    return prob_map.reshape(H, W, n_classes)

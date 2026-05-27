import warnings
from contextlib import contextmanager

import numpy as np
import zarr
import zarr.storage

warnings.filterwarnings(
    "ignore",
    message=".*LMDBStore is deprecated.*",
    category=FutureWarning,
)


@contextmanager
def open_pred_store(path: str, map_size_tb: float = 1.0):
    store = zarr.LMDBStore(path, map_size=int(map_size_tb * 1e12))
    grp = zarr.open_group(store, mode="a")
    try:
        yield grp
    finally:
        store.close()  # guaranteed even if inference crashes mid-loop


def _chunk_for(H: int, W: int, max_mb: float = 8.0) -> tuple[int, int]:
    """
    Return chunk shape that fits within max_mb and divides evenly into (H, W).
    Images are 2^k x 2^l so we just halve until we fit.
    """
    ch, cw = H, W
    while (ch * cw * 2) / 1e6 > max_mb:  # float16 = 2 bytes
        if ch >= cw:
            ch //= 2
        else:
            cw //= 2
    return (ch, cw)


def save_inference_outputs_zarr(
    prob_map: np.ndarray,  # (H, W, n_classes) float16
    image_name: str,
    store: zarr.Group,
    N: int,
    seed: int,
    background_idx: int = 0,
    top_k_save: int = 3,
    true_idx: int | None = None,
    hparams: dict | None = None,
):
    H, W, n_classes = prob_map.shape
    trial_id = f"N{N}_seed{seed:02d}"
    compressor = zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    chunk_hw = _chunk_for(H, W)
    chunk_hwk = (*chunk_hw, top_k_save)

    img_grp = store.require_group(f"{trial_id}/{image_name}")
    img_grp.attrs.update(
        {
            "N": N,
            "seed": seed,
            "trial_id": trial_id,
            "true_idx": true_idx,
            "background_idx": background_idx,
            "n_classes": n_classes,
            "top_k_saved": top_k_save,
            "H": H,
            "W": W,
            **(hparams or {}),
        }
    )

    img_grp.array(
        "argmax_map",
        np.argmax(prob_map, axis=-1).astype(np.uint8),
        dtype="u1",
        overwrite=True,
        chunks=chunk_hw,
        compressor=compressor,
    )

    img_grp.array(
        "bg_prob",
        prob_map[:, :, background_idx].astype(np.float16),
        dtype="float16",
        overwrite=True,
        chunks=chunk_hw,
        compressor=compressor,
    )

    top_k_save = min(top_k_save, n_classes)
    flat = prob_map.reshape(-1, n_classes).astype(np.float32)
    top_k_idx = np.argsort(flat, axis=-1)[:, -top_k_save:][:, ::-1]
    top_k_prob = flat[np.arange(flat.shape[0])[:, None], top_k_idx]

    img_grp.array(
        "top_k_classes",
        top_k_idx.reshape(H, W, top_k_save).astype(np.uint8),
        dtype="u1",
        overwrite=True,
        chunks=chunk_hwk,
        compressor=compressor,
    )

    img_grp.array(
        "top_k_probs",
        top_k_prob.reshape(H, W, top_k_save).astype(np.float16),
        dtype="float16",
        overwrite=True,
        chunks=chunk_hwk,
        compressor=compressor,
    )

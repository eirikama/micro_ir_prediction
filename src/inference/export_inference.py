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
    prob_map: np.ndarray,        # (H, W, n_classes) spatial  OR  (N, n_classes) flat
    image_name: str,
    store: zarr.Group,
    N: int,
    seed: int,
    true_idx: int | np.ndarray,
    background_idx: int = 0,
    top_k_save: int = 3,
    aug_summary: dict | None = None,
    hparams: dict | None = None,
):
    aug_summary = aug_summary or {}
    if aug_summary.get("augment_train", False):
        enabled = sorted([
            k for k, v in aug_summary.items()
            if isinstance(v, dict) and v.get("enabled", False)
        ])
        aug_tag = "+".join(enabled) if enabled else "aug_none"
    else:
        aug_tag = "raw"
    trial_id = f"N{N}_seed{seed:02d}_{aug_tag}"

    compressor = zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)

    # ── detect layout ─────────────────────────────────────────────────────────
    # run_inference returns (H, W, n_classes) — for PCUK W=1 (faked cube)
    # squeeze that to (N, n_classes) flat
    is_flat = (prob_map.ndim == 3 and prob_map.shape[1] == 1) or prob_map.ndim == 2
    if is_flat:
        prob_map   = prob_map.reshape(-1, prob_map.shape[-1])  # (N, n_classes)
        n_pixels, n_classes = prob_map.shape
        _save_flat(
            prob_map, image_name, store, trial_id,
            true_idx, background_idx, top_k_save,
            n_pixels, n_classes, compressor,
            N, seed, aug_summary, hparams,
        )
    else:
        H, W, n_classes = prob_map.shape
        _save_spatial(
            prob_map, image_name, store, trial_id,
            true_idx, background_idx, top_k_save,
            H, W, n_classes, compressor,
            N, seed, aug_summary, hparams,
        )


def _save_spatial(
    prob_map, image_name, store, trial_id,
    true_idx, background_idx, top_k_save,
    H, W, n_classes, compressor,
    N, seed, aug_summary, hparams,
):
    """Original spatial save path — hyperspectral image (H, W, n_classes)."""
    chunk_hw  = _chunk_for(H, W)
    chunk_hwk = (*chunk_hw, min(top_k_save, n_classes))

    img_grp = store.require_group(f"{trial_id}/{image_name}")

    # Write the data arrays first, attrs last (with a "complete" marker).
    # A group's attrs — including "layout" — used to be written before its
    # arrays, so a hard crash mid-save (OOM kill, worker crash) could leave
    # a group that *looks* valid (has "layout") but is missing the arrays a
    # reader expects, raising a bare KeyError far later during analysis.
    # Writing arrays first means a crash before attrs are set leaves a group
    # with no attrs at all — readers should treat that (or a missing/False
    # "complete" attr) as incomplete and skip it.
    img_grp.array("argmax_map",
        np.argmax(prob_map, axis=-1).astype(np.uint8),
        dtype="u1", overwrite=True, chunks=chunk_hw, compressor=compressor)

    img_grp.array("bg_prob",
        prob_map[:, :, background_idx].astype(np.float16),
        dtype="float16", overwrite=True, chunks=chunk_hw, compressor=compressor)

    top_k = min(top_k_save, n_classes)
    flat      = prob_map.reshape(-1, n_classes).astype(np.float32)
    top_k_idx  = np.argsort(flat, axis=-1)[:, -top_k:][:, ::-1]
    top_k_prob = flat[np.arange(flat.shape[0])[:, None], top_k_idx]

    img_grp.array("top_k_classes",
        top_k_idx.reshape(H, W, top_k).astype(np.uint8),
        dtype="u1", overwrite=True, chunks=chunk_hwk, compressor=compressor)

    img_grp.array("top_k_probs",
        top_k_prob.reshape(H, W, top_k).astype(np.float16),
        dtype="float16", overwrite=True, chunks=chunk_hwk, compressor=compressor)

    img_grp.attrs.update({
        "true_idx":       true_idx.tolist() if isinstance(true_idx, np.ndarray) else int(true_idx),
        "N": N, "seed": seed, "trial_id": trial_id,
        "background_idx": background_idx,
        "n_classes":      n_classes,
        "top_k_saved":    top_k_save,
        "H": H, "W": W,
        "layout":         "spatial",
        "aug":            aug_summary or {},
        **(hparams or {}),
        "complete":       True,
    })


def _save_flat(
    prob_map, image_name, store, trial_id,
    true_idx, background_idx, top_k_save,
    n_pixels, n_classes, compressor,
    N, seed, aug_summary, hparams,
):
    """Flat save path — annotated spectra (N, n_classes), e.g. PCUK cores."""
    top_k = min(top_k_save, n_classes)
    chunk  = (min(n_pixels, 4096),)
    chunkk = (min(n_pixels, 4096), top_k)

    img_grp = store.require_group(f"{trial_id}/{image_name}")

    # Arrays first, attrs (with "complete") last — see the note in
    # _save_spatial for why: it makes a crash mid-save leave a group a
    # reader can recognize as incomplete instead of one that has "layout"
    # set but is silently missing "argmax"/"bg_prob"/etc.
    img_grp.array("argmax",
        np.argmax(prob_map, axis=-1).astype(np.uint8),
        dtype="u1", overwrite=True, chunks=chunk, compressor=compressor)

    img_grp.array("bg_prob",
        prob_map[:, background_idx].astype(np.float16),
        dtype="float16", overwrite=True, chunks=chunk, compressor=compressor)

    # true labels — only meaningful for flat layout where we have per-pixel GT
    if isinstance(true_idx, np.ndarray):
        img_grp.array("true_labels",
            true_idx.astype(np.uint8),
            dtype="u1", overwrite=True, chunks=chunk, compressor=compressor)

    top_k_idx  = np.argsort(prob_map, axis=-1)[:, -top_k:][:, ::-1].astype(np.uint8)
    top_k_prob = prob_map[np.arange(n_pixels)[:, None], top_k_idx].astype(np.float16)

    img_grp.array("top_k_classes",
        top_k_idx, dtype="u1", overwrite=True, chunks=chunkk, compressor=compressor)

    img_grp.array("top_k_probs",
        top_k_prob, dtype="float16", overwrite=True, chunks=chunkk, compressor=compressor)

    img_grp.attrs.update({
        "true_idx":       true_idx.tolist() if isinstance(true_idx, np.ndarray) else int(true_idx),
        "N": N, "seed": seed, "trial_id": trial_id,
        "background_idx": background_idx,
        "n_classes":      n_classes,
        "top_k_saved":    top_k,
        "n_pixels":       n_pixels,
        "layout":         "flat",
        "aug":            aug_summary or {},
        **(hparams or {}),
        "complete":       True,
    })

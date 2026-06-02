"""
Standalone inference — no Hydra, no MLflow required.

Two input modes
---------------
zarr store (multi-GPU, lazy loading):
    python predict.py --ckpt checkpoints/best.ckpt \\
                      --zarr /data/test.zarr --image img_001

numpy / npz file (single GPU, array already in memory):
    python predict.py --ckpt checkpoints/best.ckpt --npy /data/my_image.npy
    python predict.py --ckpt checkpoints/best.ckpt --npy /data/spectra.npz --npy-key data

Explore a zarr store:
    python predict.py --zarr /data/test.zarr --list

Output
------
<out>/<stem>.npz  with arrays:
  prob_map  (H, W, n_classes) or (N, n_classes)  float16
  argmax    (H, W) or (N,)                        uint8
  bg_prob   (H, W) or (N,)                        float16  [background = class 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch.multiprocessing as mp


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_images(zarr_path: str) -> None:
    import zarr
    store = zarr.open(zarr_path, mode="r")
    if "images" in store:
        keys = list(store["images"].keys())
        print(f"Found {len(keys)} images in {zarr_path}:")
        for k in sorted(keys):
            label = store["images"][k].attrs.get("label", "—")
            print(f"  {k:<40}  label={label}")
    else:
        keys = list(store.group_keys())
        print(f"Found {len(keys)} groups in {zarr_path}:")
        for k in sorted(keys):
            print(f"  {k}")


def _load_array(path: str, key: str | None = None) -> np.ndarray:
    """Load a .npy or .npz file into a numpy array.

    For .npz, the array key is resolved in this order:
      1. ``--npy-key`` if given
      2. First of the known default keys: spectra, data, X, image, arr_0
      3. The first key in the archive
    """
    p = Path(path)
    if p.suffix == ".npz":
        archive = np.load(path)
        if key and key in archive:
            return archive[key]
        for default in ("spectra", "data", "X", "image", "arr_0"):
            if default in archive:
                return archive[default]
        first = list(archive.keys())[0]
        print(
            f"Warning: no known key found in {p.name}; using first key '{first}'.",
            file=sys.stderr,
        )
        return archive[first]
    return np.load(path)


def _print_result(prob_map: np.ndarray, out_path: Path) -> None:
    """Save output and print a summary."""
    is_image = prob_map.ndim == 3
    if is_image:
        H, W, n_classes = prob_map.shape
        flat = prob_map.reshape(-1, n_classes)
    else:
        flat = prob_map
        n_classes = prob_map.shape[1]

    argmax  = flat.argmax(-1).astype(np.uint8)
    bg_prob = flat[:, 0].astype(np.float16)

    if is_image:
        argmax  = argmax.reshape(H, W)
        bg_prob = bg_prob.reshape(H, W)

    np.savez_compressed(
        out_path,
        prob_map=prob_map,
        argmax=argmax,
        bg_prob=bg_prob,
    )

    print(f"Saved  → {out_path}")
    print(f"  prob_map : {prob_map.shape}  dtype={prob_map.dtype}")
    print(f"  argmax   : {argmax.shape}")
    print()
    for cls_idx in np.unique(argmax):
        n   = int((argmax == cls_idx).sum())
        pct = 100 * n / argmax.size
        print(f"  class {cls_idx:>3}: {n:>8}  ({pct:5.1f}%)")


# ── run modes ────────────────────────────────────────────────────────────────

def _run_zarr(args: argparse.Namespace) -> None:
    """Multi-GPU zarr-backed inference."""
    from src.inference.inference_engine import run_inference

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.image}.npz"

    print(f"Mode       : zarr")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Zarr store : {args.zarr}")
    print(f"Image      : {args.image}")
    print(f"GPUs       : {args.devices}")
    print(f"Batch size : {args.batch_size}")
    print()

    prob_map = run_inference(
        image_name=args.image,
        ckpt_path=args.ckpt,
        zarr_path=args.zarr,
        batch_size=args.batch_size,
        devices=args.devices,
    )
    _print_result(prob_map, out_path)


def _run_npy(args: argparse.Namespace) -> None:
    """Single-GPU inference on a numpy / npz file."""
    from src.inference.inference_engine import predict_array

    spectra  = _load_array(args.npy, args.npy_key)
    stem     = Path(args.npy).stem
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.npz"

    print(f"Mode        : numpy array")
    print(f"Input       : {args.npy}  shape={spectra.shape}  dtype={spectra.dtype}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Device      : {args.devices[0]}")
    print(f"Batch size  : {args.batch_size}")
    print(f"z-normalize : {args.z_normalize}")
    print()

    prob_map = predict_array(
        spectra     = spectra,
        ckpt_path   = args.ckpt,
        batch_size  = args.batch_size,
        device      = args.devices[0],
        z_normalize = args.z_normalize,
    )
    _print_result(prob_map, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone inference — no Hydra or MLflow needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # shared
    p.add_argument("--ckpt",        help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--out",         default="predictions",
                   help="Output directory (default: predictions/)")
    p.add_argument("--devices",     type=int, nargs="+", default=[0],
                   help="GPU ids (default: 0).  zarr mode uses all listed; "
                        "numpy mode uses only the first.")
    p.add_argument("--batch-size",  type=int, default=512, dest="batch_size",
                   help="Spectra per GPU batch (default: 512)")

    # zarr mode
    zarr_grp = p.add_argument_group("zarr input")
    zarr_grp.add_argument("--zarr",  help="Path to zarr store")
    zarr_grp.add_argument("--image", help="Image key inside the zarr store")
    zarr_grp.add_argument("--list",  action="store_true",
                          help="List images in --zarr and exit")

    # numpy mode
    npy_grp = p.add_argument_group("numpy / npz input")
    npy_grp.add_argument("--npy",       metavar="PATH",
                         help=".npy or .npz file containing the spectra array. "
                              "Shape: (H, W, L) or (N, L).")
    npy_grp.add_argument("--npy-key",   metavar="KEY", dest="npy_key", default=None,
                         help="Array key inside a .npz archive (auto-detected if omitted)")
    npy_grp.add_argument("--z-normalize", action="store_true", dest="z_normalize",
                         help="Apply per-spectrum z-score normalisation before inference. "
                              "Enable if data.z_normalize=True was used during training.")

    return p


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args   = _build_parser().parse_args()
    has_zarr = bool(args.zarr)
    has_npy  = bool(args.npy)

    # ── list mode ─────────────────────────────────────────────────────────────
    if args.list:
        if not has_zarr:
            print("error: --zarr is required with --list", file=sys.stderr)
            sys.exit(1)
        _list_images(args.zarr)
        sys.exit(0)

    # ── validate ──────────────────────────────────────────────────────────────
    if not args.ckpt:
        print("error: --ckpt is required", file=sys.stderr)
        sys.exit(1)
    if not has_zarr and not has_npy:
        print("error: provide either --zarr --image  or  --npy", file=sys.stderr)
        sys.exit(1)
    if has_zarr and has_npy:
        print("error: --zarr and --npy are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if has_zarr and not args.image:
        print("error: --image is required with --zarr", file=sys.stderr)
        sys.exit(1)

    # ── dispatch ──────────────────────────────────────────────────────────────
    if has_npy:
        _run_npy(args)
    else:
        _run_zarr(args)

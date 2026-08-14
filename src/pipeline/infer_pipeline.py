"""Inference pipeline — per-image inference loop, accuracy logging, LMDB saving."""
from __future__ import annotations

import logging
import time
import traceback
from collections import defaultdict

import numpy as np
import zarr
from omegaconf import DictConfig

from src.inference.inference_engine import InferenceSession, resolve_image_group
from src.inference.export_inference import open_pred_store, save_inference_outputs_zarr
from src.tracking.base import Tracker

log = logging.getLogger(__name__)


def run_inference_pipeline(
    cfg: DictConfig,
    test_images: list[dict],
    label_encoding: dict[str, int],
    aug_summary_dict: dict,
    tracker: Tracker,
) -> float | None:

    image_metrics: list[dict] = []
    failed:        list[str]  = []

    # True for "flat" annotated-spectra stores (PCUK, bacteria, milk, ...);
    # False for spatial hyperspectral cubes (microplastics, pollen, ...).
    is_flat_layout = not cfg.data.hyperspectra
    log.debug("is_flat_layout=%s", is_flat_layout)

    # open the right store for ground truth
    if is_flat_layout:
        root_gt = zarr.open(
            cfg.data.zarr_path
            if cfg.data.intrinsic_validation
            else cfg.data.zarr_test_path,
            mode="r",
        )
    else:
        root_test = (
            zarr.open(cfg.data.zarr_test_path, mode="r")
            if not cfg.data.intrinsic_validation
            else None
        )

    log.info("Starting inference on %d images...", len(test_images))
    t_infer_start = time.time()

    # One model load (or one persistent worker pool for multi-GPU) reused
    # across every image below — the checkpoint used to be reloaded from
    # disk per image (and per GPU worker), which dominated inference time.
    inference_zarr_path = (
        cfg.data.zarr_path
        if cfg.data.intrinsic_validation
        else cfg.data.zarr_test_path
    )

    with open_pred_store(cfg.inference.pred_store_path) as pred_store, \
         InferenceSession(
             ckpt_path=cfg.inference.ckpt_path,
             devices=list(cfg.inference.devices),
             batch_size=cfg.inference.batch_size,
         ) as inference_session:
        for i, img_data in enumerate(test_images):
            img_name  = img_data["name"]
            img_label = img_data["label"]
            safe_label = img_label.replace(" ", "_").replace("/", "_")

            log.info("Inference [%d/%d]: %s", i + 1, len(test_images), img_name)

            try:
                t0 = time.time()
                prob_map = inference_session.infer(inference_zarr_path, img_name)

                # ── ground truth ──────────────────────────────────────────
                # Neither is_flat_layout nor intrinsic_validation reliably
                # predicts whether a store carries per-pixel "y" labels, or
                # even where a given image's group lives in the store (see
                # resolve_image_group) — different domains have shown up
                # with every combination. So just look at what's actually
                # there: PCUK/bacteria tissue cores have "y"; milk is
                # single-label-per-sample and doesn't, so it falls back to
                # the label_encoding lookup like other intrinsic datasets.
                if is_flat_layout:
                    gt_grp       = resolve_image_group(root_gt, img_name)
                    has_pixel_gt = "y" in gt_grp
                    y_true       = gt_grp["y"][:] if has_pixel_gt else label_encoding.get(img_label)
                else:
                    has_pixel_gt = False
                    if cfg.data.intrinsic_validation:
                        y_true = label_encoding.get(img_label)    # single int
                    else:
                        y_true = root_test[img_name]["y"][:]

                save_inference_outputs_zarr(
                    prob_map       = prob_map,
                    image_name     = img_name,
                    store          = pred_store,
                    N              = cfg.data.spectra_per_class,
                    seed           = cfg.seed,
                    true_idx       = y_true,
                    background_idx = cfg.inference.background_idx,
                    top_k_save     = cfg.inference.top_k_save,
                    aug_summary    = aug_summary_dict,
                    hparams        = {
                        "lr":         cfg.model.lr,
                        "batch_size": cfg.data.batch_size,
                        "mlflow_run": tracker.run_id,
                    },
                )

                argmax = prob_map.reshape(-1, prob_map.shape[-1]).argmax(-1)

                # ── accuracy logging ──────────────────────────────────────
                if has_pixel_gt:
                    # per-class accuracy using per-pixel GT
                    acc_overall = float(np.mean(argmax == y_true))
                    for class_name, class_idx in label_encoding.items():
                        mask = y_true == class_idx
                        if not mask.any():
                            continue
                        acc_cls  = float(np.mean(argmax[mask] == class_idx))
                        safe_cls = f"{img_name}_{class_name}".replace(" ", "_")
                        image_metrics.append({"image": safe_cls, "accuracy": acc_cls})
                        tracker.log_metric(
                            f"Inference/per_class/acc_{safe_cls}", acc_cls, step=i
                        )
                    image_metrics.append({"image": img_name, "accuracy": acc_overall})
                    tracker.log_metric(
                        f"Inference/acc_{img_name}", acc_overall, step=i
                    )
                    acc = acc_overall

                elif not is_flat_layout and not cfg.data.intrinsic_validation:
                    # original spatial path — one label per image
                    acc_overall  = float(np.mean(argmax == y_true))
                    class_counts = dict(root_test[img_name].attrs["class_counts"])
                    for class_name in class_counts:
                        class_idx = label_encoding.get(class_name)
                        if class_idx is None:
                            continue
                        mask = y_true == class_idx
                        if not mask.any():
                            continue
                        acc_cls  = float(np.mean(argmax[mask] == class_idx))
                        safe_cls = f"{img_name}_{class_name}".replace(" ", "_")
                        image_metrics.append({"image": safe_cls, "accuracy": acc_cls})
                        tracker.log_metric(
                            f"Inference/per_class/acc_{safe_cls}", acc_cls, step=i
                        )
                    image_metrics.append({"image": img_name, "accuracy": acc_overall})
                    tracker.log_metric(
                        f"Inference/acc_{img_name}", acc_overall, step=i
                    )
                    acc = acc_overall

                else:
                    # intrinsic validation — single label per image
                    bg_prob = prob_map[:, :, cfg.inference.background_idx].flatten()
                    mask    = bg_prob <= cfg.inference.bg_threshold
                    acc     = (
                        float(np.mean(argmax[mask] == y_true))
                        if mask.sum() > 0
                        else float("nan")
                    )
                    image_metrics.append({"image": img_label, "accuracy": acc})
                    tracker.log_metric(
                        f"Inference/per_class/acc_{safe_label}", acc, step=i
                    )

                log.info("  %s accuracy: %.4f", img_label, acc)
                tracker.log_metric(
                    f"Timer/per_class/infer_min_{safe_label}",
                    (time.time() - t0) / 60,
                )

            except Exception as e:
                tracker.log_text(traceback.format_exc(), f"errors/infer_{safe_label}.txt")
                tracker.set_tag("error", str(e))
                failed.append(img_name)
                log.error("Failed image %s: %s", img_name, e, exc_info=True)

    # ── summary metrics (unchanged) ───────────────────────────────────────────
    if image_metrics:
        accs = [m["accuracy"] for m in image_metrics if not np.isnan(m["accuracy"])]
        if accs:
            class_accs: dict = defaultdict(list)
            for m in image_metrics:
                if not np.isnan(m["accuracy"]):
                    class_accs[m["image"]].append(m["accuracy"])
            for cls, cls_accs in class_accs.items():
                safe = cls.replace(" ", "_").replace("/", "_")
                tracker.log_metric(
                    f"Inference/per_class/mean_acc_{safe}",
                    float(np.mean(cls_accs)),
                )
            tracker.log_metrics({
                "Inference/mean_accuracy":   float(np.mean(accs)),
                "Inference/min_accuracy":    float(np.min(accs)),
                "Inference/max_accuracy":    float(np.max(accs)),
                "Inference/std_accuracy":    float(np.std(accs)),
                "Inference/n_images_ok":     len(accs),
                "Inference/n_images_failed": len(failed),
            })

    tracker.log_metric("Timer/total_infer_min", (time.time() - t_infer_start) / 60)
    tracker.set_tag("pred_store_path", cfg.inference.pred_store_path)

    if failed:
        tracker.log_text("\n".join(failed), "failed_images.txt")
        log.warning("Failed images: %s", failed)

    log.info(
        "Done. %d/%d images saved.",
        len(test_images) - len(failed),
        len(test_images),
    )

    if image_metrics:
        accs = [m["accuracy"] for m in image_metrics if not np.isnan(m["accuracy"])]
        if accs:
            return float(np.mean(accs))
    return None

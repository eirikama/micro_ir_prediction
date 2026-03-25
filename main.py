import gc
import logging
import platform
import time
from collections import defaultdict

import hydra
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import MLFlowLogger

from config_schema import DataConfig, ModelConfig, TrainerConfig
from src.Data.data_utils import SpectralDataModule, create_experiment_split
from src.Engine.inference_engine import run_inference
from src.Engine.save_inference import open_pred_store, save_inference_outputs_zarr
from src.Engine.trainer_engine import run_training
from src.Models.aacnn import AACNN

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(group="model", name="aacnn_config", node=ModelConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="trainer", name="trainer_config", node=TrainerConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    model_cfg = hydra.utils.instantiate(cfg.model)
    data_cfg = hydra.utils.instantiate(cfg.data)

    train_test_split = create_experiment_split(data_cfg.zarr_path, split_ratio=0.9)

    datamodule = SpectralDataModule(train_test_split["train"], data_cfg)
    datamodule.setup()
    label_encoding = datamodule.label_encoding

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"N{cfg.data.spectra_per_plastic}_seed{cfg.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        try:
            run_id = run.info.run_id
            log.info("MLflow run started: %s  run_id: %s", run_name, run_id)

            mlflow.log_params(
                {
                    "N": cfg.data.spectra_per_plastic,
                    "seed": cfg.seed,
                    "lr": cfg.model.lr,
                    "batch_size": cfg.data.batch_size,
                    "max_epochs": cfg.trainer.max_epochs,
                    "mode": cfg.mode,
                    "bg_threshold": cfg.bg_threshold,
                    "background_idx": cfg.background_idx,
                    "top_k_save": cfg.top_k_save,
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "cpu",
                    "gpu_count": torch.cuda.device_count(),
                    "hostname": platform.node(),
                }
            )

            mlflow.log_dict(label_encoding, "label_encoding.json")

            mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")
            mlflow.set_tags(
                {
                    "mode": cfg.mode,
                    "N": str(cfg.data.spectra_per_plastic),
                    "seed": str(cfg.seed),
                }
            )

            # ── train ─────────────────────────────────────────────────────────
            if cfg.mode in ["train", "all"]:
                model = AACNN(model_cfg)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                log.info("Trainable parameters: %d", n_params)

                mlf_logger = MLFlowLogger(
                    experiment_name=cfg.mlflow.experiment_name,
                    tracking_uri=cfg.mlflow.tracking_uri,
                    run_id=run_id,
                )

                t0 = time.time()

                best_path, best_score, final_epoch, stopped_early = run_training(
                    cfg, model, datamodule, mlf_logger
                )

                train_sec = time.time() - t0
                mlflow.log_metric("Timer/train_duration_min", train_sec / 60)

                mlflow.log_metrics(
                    {
                        "best_val_score": best_score,
                        "final_epoch": final_epoch,
                    }
                )
                mlflow.set_tag("stopped_early", str(stopped_early))

                with open_dict(cfg):
                    cfg.ckpt_path = best_path

                mlflow.set_tag("ckpt_path", best_path)
                log.info("Training complete. Best checkpoint: %s", best_path)

                del model
                torch.cuda.empty_cache()
                gc.collect()

            ################# infer ##################

            if cfg.mode in ["infer", "all"]:
                if not cfg.ckpt_path:
                    raise ValueError("ckpt_path must be set to run inference.")

                test_images = train_test_split["test"]
                image_metrics = []
                failed = []

                log.info("Starting inference on %d images...", len(test_images))

                t_infer_start = time.time()
                with open_pred_store(cfg.pred_store_path) as pred_store:
                    for i, img_data in enumerate(test_images):
                        img_name = img_data["name"]
                        img_label = img_data["label"]
                        true_idx = label_encoding[img_label]
                        safe_label = img_label.replace(" ", "_").replace("/", "_")

                        log.info("Inference [%d/%d]: %s", i + 1, len(test_images), img_name)

                        try:
                            t0 = time.time()
                            prob_map = run_inference(
                                cfg,
                                image_name=img_name,
                                ckpt_path=cfg.ckpt_path,
                            )

                            save_inference_outputs_zarr(
                                prob_map=prob_map,
                                image_name=img_name,
                                store=pred_store,
                                N=cfg.data.spectra_per_plastic,
                                seed=cfg.seed,
                                background_idx=cfg.background_idx,
                                top_k_save=cfg.top_k_save,
                                true_idx=true_idx,
                                hparams={
                                    "lr": cfg.model.lr,
                                    "batch_size": cfg.data.batch_size,
                                    "mlflow_run": run_id,
                                },
                            )

                            bg_prob = prob_map[:, :, cfg.background_idx].flatten()
                            argmax = prob_map.reshape(-1, prob_map.shape[-1]).argmax(-1)
                            mask = bg_prob <= cfg.bg_threshold
                            acc = (
                                float(np.mean(argmax[mask] == true_idx))
                                if mask.sum() > 0
                                else float("nan")
                            )

                            image_metrics.append({"image": img_label, "accuracy": acc})
                            mlflow.log_metric(f"Inference/acc_{safe_label}", acc, step=i)

                            log.info("  %s accuracy: %.4f", img_label, acc)
                            mlflow.log_metric(
                                f"Timer/infer_min_{safe_label}", (time.time() - t0) / 60
                            )

                        except Exception as e:
                            log.warning("Inference failed for %s: %s", img_name, e)
                            failed.append(img_name)

                if image_metrics:
                    accs = [m["accuracy"] for m in image_metrics if not np.isnan(m["accuracy"])]
                    if accs:
                        class_accs = defaultdict(list)
                        for m in image_metrics:
                            if not np.isnan(m["accuracy"]):
                                class_accs[m["image"]].append(m["accuracy"])

                        for cls, cls_accs in class_accs.items():
                            safe = cls.replace(" ", "_").replace("/", "_")
                            mlflow.log_metric(
                                f"Inference/mean_acc_{safe}", float(np.mean(cls_accs))
                            )

                        mlflow.log_metrics(
                            {
                                "Inference/mean_accuracy": float(np.mean(accs)),
                                "Inference/min_accuracy": float(np.min(accs)),
                                "Inference/max_accuracy": float(np.max(accs)),
                                "Inference/std_accuracy": float(np.std(accs)),
                                "Inference/n_images_ok": len(accs),
                                "Inference/n_images_failed": len(failed),
                            }
                        )

                mlflow.log_metric("Timer/total_infer_min", (time.time() - t_infer_start) / 60)
                mlflow.set_tag("pred_store_path", cfg.pred_store_path)

                if failed:
                    mlflow.log_text("\n".join(failed), "failed_images.txt")
                    log.warning("Failed images: %s", failed)

                log.info(
                    "Done. %d/%d images saved.",
                    len(test_images) - len(failed),
                    len(test_images),
                )

            mlflow.set_tag("status", "completed")
        except Exception as e:
            mlflow.set_tag("error", str(e))
            raise


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

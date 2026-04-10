import gc
import getpass
import logging
import platform
import time
from collections import defaultdict
import git

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
from src.utils.warnings import silence_warnings

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(group="model", name="aacnn_config", node=ModelConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="trainer", name="trainer_config", node=TrainerConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    silence_warnings()
    pl.seed_everything(cfg.seed)

    model_cfg = hydra.utils.instantiate(cfg.model)
    data_cfg = hydra.utils.instantiate(cfg.data)

    train_test_split = create_experiment_split(data_cfg.zarr_path, split_ratio=0.5)

    datamodule = SpectralDataModule(train_test_split["train"], data_cfg)
    datamodule.setup()
    label_encoding = datamodule.label_encoding

    run_name = f"N{cfg.data.spectra_per_plastic}_seed{cfg.seed}"

    repo_path = hydra.utils.get_original_cwd()
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        is_dirty = repo.is_dirty()
        # Capture uncommitted changes
        git_diff = repo.git.diff(repo.head.commit) if is_dirty else "No uncommitted changes."
    except Exception as e:
        log.warning(f"Could not capture git state: {e}")
        commit_hash, is_dirty, git_diff = "unknown", None, "N/A"

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=run_name,
    )

    with mlflow.start_run(run_id=mlf_logger.run_id) as run:
        try:
            log.info("MLflow run started: %s | ID: %s", run_name, run.info.run_id)

            params = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(params)
            mlflow.set_tags({
                "git.commit": commit_hash,
                "git.is_dirty": str(is_dirty),
                "git.branch": repo.active_branch.name if commit_hash != "unknown" else "N/A"
            })

            mlflow.log_text(git_diff, "git_patch.diff")

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

            mlflow.set_tags(
                {
                    "mlflow_id": run.info.run_id,
                    "mlflow_name": run_name,
                    "mode": cfg.mode,
                    "spectra_per_plastic": cfg.data.spectra_per_plastic,
                    "user": getpass.getuser(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "gpu_count": gpu_count,
                    "all_gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                    if gpu_count > 0
                    else "N/A",
                    "Hardware/gpu_memory_GB": torch.cuda.get_device_properties(0).total_memory
                    / 1e9,
                    "hostname": platform.node(),
                    "status": "running",
                }
            )

            mlflow.log_dict(label_encoding, "label_encoding.json")
            mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

            # ── train ─────────────────────────────────────────────────────────
            if cfg.mode in ["train", "all"]:
                model = AACNN(model_cfg)

                mlflow.log_params(
                    {
                        "Model/n_params_trainable": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "Model/n_params_total": sum(p.numel() for p in model.parameters()),
                    }
                )

                t0 = time.time()

                best_path, best_score, final_epoch, stopped_early = run_training(
                    cfg, model, datamodule, mlf_logger
                )

                train_seconds = time.time() - t0

                mlflow.log_metrics(
                    {
                        "Timer/train_duration_minutes": train_seconds / 60,
                        "Timer/epochs_per_second": final_epoch / train_seconds,
                        "Timer/samples_per_second": final_epoch
                        * cfg.data.batch_size
                        / train_seconds,
                        "Accuracy/best_val_acc": best_score,
                        "Trainer/final_epoch": final_epoch,
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
                                    "mlflow_run": run.info.run_id,
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

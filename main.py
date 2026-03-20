import logging
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow

from config_schema import ModelConfig, DataConfig, TrainerConfig
from src.Models.aacnn import AACNN
from src.Data.data_utils import SpectralDataModule, create_experiment_split
from src.Engine.trainer_engine import run_training
from src.Engine.inference_engine import run_inference
from src.Engine.save_inference import save_inference_outputs_zarr, open_pred_store

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(group="model",   name="aacnn_config",   node=ModelConfig)
cs.store(group="data",    name="data_config",    node=DataConfig)
cs.store(group="trainer", name="trainer_config", node=TrainerConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    model_cfg = hydra.utils.instantiate(cfg.model)
    data_cfg  = hydra.utils.instantiate(cfg.data)

    train_test_split = create_experiment_split(data_cfg.zarr_path, split_ratio=0.9)

    # always build — label_encoding needed in both train and infer
    datamodule     = SpectralDataModule(train_test_split["train"], data_cfg)
    datamodule.setup()
    label_encoding = datamodule.label_encoding

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"N{cfg.data.spectra_per_plastic}_seed{cfg.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run started: %s  run_id: %s", run_name, run_id)

        mlflow.log_params({
            "N":          cfg.data.spectra_per_plastic,
            "seed":       cfg.seed,
            "lr":         cfg.model.lr,
            "batch_size": cfg.data.batch_size,
            "max_epochs": cfg.trainer.max_epochs,
            "mode":       cfg.mode,
        })
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")
        mlflow.set_tags({
            "mode": cfg.mode,
            "N":    str(cfg.data.spectra_per_plastic),
            "seed": str(cfg.seed),
        })

        # ── train ─────────────────────────────────────────────────────────────
        if cfg.mode in ["train", "all"]:

            model = AACNN(model_cfg)

            mlf_logger = MLFlowLogger(
                experiment_name = cfg.mlflow.experiment_name,
                tracking_uri    = cfg.mlflow.tracking_uri,
                run_id          = run_id,
            )

            best_path     = run_training(cfg, model, datamodule, mlf_logger)
            cfg.ckpt_path = best_path

            mlflow.log_artifact(best_path, artifact_path="checkpoints")
            mlflow.set_tag("ckpt_path", best_path)
            log.info("Training complete. Best checkpoint: %s", best_path)

            
            del model
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        if cfg.mode in ["infer", "all"]:
            if not cfg.ckpt_path:
                raise ValueError("ckpt_path must be set to run inference.")

            test_images   = train_test_split["test"]
            image_metrics = []   
            failed        = []

            log.info("Starting inference on %d images...", len(test_images))

            with open_pred_store(cfg.pred_store_path) as pred_store:
                for i, img_data in enumerate(test_images):
                    img_name  = img_data["name"]
                    img_label = img_data["label"]
                    true_idx  = label_encoding[img_label]

                    log.info("Inference [%d/%d]: %s", i + 1, len(test_images), img_name)

                    try:
                        prob_map = run_inference(
                            cfg,
                            image_name = img_name,
                            ckpt_path  = cfg.ckpt_path,
                        )

                        save_inference_outputs_zarr(
                            prob_map       = prob_map,
                            image_name     = img_name,
                            store          = pred_store,
                            N              = cfg.data.spectra_per_plastic,
                            seed           = cfg.seed,
                            background_idx = cfg.background_idx,
                            top_k_save     = cfg.top_k_save,
                            true_idx       = true_idx,
                            hparams        = {
                                "lr":         cfg.model.lr,
                                "batch_size": cfg.data.batch_size,
                                "mlflow_run": run_id,
                            },
                        )

                        bg_prob = prob_map[:, :, cfg.background_idx].flatten()
                        argmax  = prob_map.reshape(-1, prob_map.shape[-1]).argmax(-1)
                        mask    = bg_prob <= 0.5
                        acc     = float(np.mean(argmax[mask] == true_idx)) if mask.sum() > 0 else float("nan")

                        image_metrics.append({"image": img_label, "accuracy": acc})
                        mlflow.log_metric(f"acc_{img_label}", acc)   # ← single quotes
                        log.info("  %s accuracy: %.4f", img_label, acc)

                    except Exception as e:
                        log.warning("Inference failed for %s: %s", img_name, e)
                        failed.append(img_name)

            if image_metrics:
                accs = [m["accuracy"] for m in image_metrics if not np.isnan(m["accuracy"])]
                mlflow.log_metrics({
                    "mean_accuracy":   float(np.mean(accs)),
                    "min_accuracy":    float(np.min(accs)),
                    "max_accuracy":    float(np.max(accs)),
                    "std_accuracy":    float(np.std(accs)),
                    "n_images_ok":     len(accs),
                    "n_images_failed": len(failed),
                })

            mlflow.set_tag("pred_store_path", cfg.pred_store_path)

            if failed:
                mlflow.log_text("\n".join(failed), "failed_images.txt")
                log.warning("Failed images: %s", failed)

            log.info(
                "Done. %d/%d images saved.",
                len(test_images) - len(failed),
                len(test_images),
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
import getpass
import logging
import platform
from collections import Counter

import git
import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict

from src.configs.config_schema import DataConfig, ModelConfig, TrainerConfig, InferenceConfig
from src.data.datamodule import SpectralDataModule
from src.data.sampling import create_experiment_split, get_test_split
from src.pipeline.train_pipeline import run_training_pipeline
from src.pipeline.infer_pipeline import run_inference_pipeline
from src.tracking import build_tracker
from src.utils import silence_warnings, setup_git

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(group="model",     name="aacnn_config",    node=ModelConfig)
cs.store(group="data",      name="data_config",     node=DataConfig)
cs.store(group="trainer",   name="trainer_config",  node=TrainerConfig)
cs.store(group="inference", name="inference_config", node=InferenceConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    setup_git()
    silence_warnings()
    pl.seed_everything(cfg.seed)

    # ── data setup ────────────────────────────────────────────────────────────
    if cfg.data.intrinsic_validation:
        train_test_split = create_experiment_split(
            cfg.data.zarr_path,
            split_ratio=cfg.data.train_split_size,
            seed=cfg.seed,
        )
        print("Train:", Counter(d["label"] for d in train_test_split["train"]))
        print("Test: ", Counter(d["label"] for d in train_test_split["test"]))
        datamodule  = SpectralDataModule(train_test_split["train"], cfg.data)
        test_images = train_test_split["test"]
    else:
        datamodule  = SpectralDataModule(None, cfg.data)
        test_images = get_test_split(cfg.data.zarr_test_path, cfg.data.zarr_path)

    datamodule.setup()
    label_encoding = datamodule.label_encoding

    # ── augmentation summary (used for trial IDs and artifact logging) ────────
    # Fluorescence fitting now happens inside datamodule.setup().
    aug_map = OmegaConf.to_container(cfg.data.get("augmentations") or {}, resolve=True)
    aug_summary_dict: dict = dict(aug_map)
    aug_summary_dict["augment_train"] = cfg.data.augment_train
    aug_summary_dict["augment_val"]   = cfg.data.augment_val

    if cfg.data.augment_train and aug_map:
        enabled = [
            aug_cfg.get("type", name)
            for name, aug_cfg in aug_map.items()
            if aug_cfg.get("enabled", True)
        ]
        aug_tag = "+".join(sorted(enabled)) if enabled else "none"
    else:
        aug_tag = "none"

    # ── git state ─────────────────────────────────────────────────────────────
    repo_path = hydra.utils.get_original_cwd()
    try:
        repo        = git.Repo(repo_path, search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        is_dirty    = repo.is_dirty()
        git_diff    = repo.git.diff(repo.head.commit) if is_dirty else "No uncommitted changes."
        git_branch  = repo.active_branch.name
    except Exception as e:
        log.warning("Could not capture git state: %s", e)
        commit_hash, is_dirty, git_diff, git_branch = "unknown", None, "N/A", "N/A"

    # ── tracking ──────────────────────────────────────────────────────────────
    run_name = f"N{cfg.data.spectra_per_class}_seed{cfg.seed}"
    tracker  = build_tracker(cfg, run_name)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    with tracker:
        log.info("Run started: %s", run_name)

        # initial params / tags / artifacts
        tracker.log_param("aug/summary", aug_tag)
        tracker.log_dict(aug_summary_dict, "augmentation_config.json")
        for aug_name, aug_info in aug_summary_dict.items():
            if isinstance(aug_info, dict):
                for k, v in aug_info.items():
                    tracker.log_param(f"aug/{aug_name}/{k}", v)
            else:
                tracker.log_param(f"aug/{aug_name}", aug_info)

        tracker.log_params(OmegaConf.to_container(cfg, resolve=True))
        tracker.set_tags({
            "git.commit":   commit_hash,
            "git.is_dirty": str(is_dirty),
            "git.branch":   git_branch,
        })
        tracker.log_text(git_diff, "git_patch.diff")
        tracker.set_tags({
            "run_id":              tracker.run_id,
            "run_name":            run_name,
            "mode":                cfg.mode,
            "spectra_per_class":   str(cfg.data.spectra_per_class),
            "user":                getpass.getuser(),
            "python_version":      platform.python_version(),
            "torch_version":       torch.__version__,
            "gpu_count":           str(gpu_count),
            "all_gpu_names":       str([torch.cuda.get_device_name(i) for i in range(gpu_count)]
                                       if gpu_count > 0 else "N/A"),
            "Hardware/gpu_memory_GB": str(
                torch.cuda.get_device_properties(0).total_memory / 1e9
            ) if gpu_count > 0 else "N/A",
            "hostname":            platform.node(),
            "status":              "running",
        })
        tracker.log_dict(label_encoding, "label_encoding.json")
        tracker.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

        best_val_acc = None

        # ── train ─────────────────────────────────────────────────────────────
        if cfg.mode in ["train", "all"]:
            best_path, best_val_acc = run_training_pipeline(cfg, datamodule, tracker)
            with open_dict(cfg):
                cfg.inference.ckpt_path = best_path

        # ── infer ─────────────────────────────────────────────────────────────
        if cfg.mode in ["infer", "all"]:
            if not cfg.inference.ckpt_path:
                raise ValueError("ckpt_path must be set to run inference.")
            run_inference_pipeline(cfg, test_images, label_encoding, aug_summary_dict, tracker)

    return best_val_acc  # used by Hydra Optuna Sweeper as the objective value


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

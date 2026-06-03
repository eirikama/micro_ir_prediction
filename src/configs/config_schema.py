from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ModelConfig:
    conv_channels: int = 64
    kernel_size: int = 21
    pred_dropout: float = 0.3
    num_classes: int = 9

    gamma: float = 1.0
    alpha: list[float] = field(default_factory=list)
    lr: float = 1e-4
    weight_decay: float = 1e-3


@dataclass
class TrainerConfig:
    Nruns: int = 10
    max_epochs: int = 500
    min_epochs: int = 5
    early_stopping_patience: int = 500
    val_every_n_epochs: int = 5
    log_every_n_epochs: int = 5
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int = 1
    gradient_clip_val: float = 1.0

@dataclass
class InferenceConfig:
    background_idx: int = 0
    bg_threshold: float = 0.5
    top_k_save: int = 3
    batch_size: int = 512
    devices: list = field(default_factory=lambda: [0, 1])
    pred_store_path: str = "/app/outputs/predictions_raw.lmdb"
    ckpt_path: Optional[str] = None


@dataclass
class DataConfig:
    zarr_path: str = ""
    zarr_test_path: str = ""
    intrinsic_validation: bool = False
    spectra_per_class: int = 8
    sample_to_bkg_spectra_ratio: int = 2
    max_sampling_per_class_attempts: int = 1000
    sampling_patch_size: int = 64
    batch_size: int = 64
    train_split_size: float = 0.5
    val_split_size: float = 0.5
    sample_min: float = 0.5
    background_max: float = 0.1
    include_bkg_pixels: bool = True
    z_normalize: bool = True
    augment_train: bool = True
    augment_val: bool = False

    modality: str = "ir"
    augmentations: Any = field(default_factory=dict)

@dataclass
class MasterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

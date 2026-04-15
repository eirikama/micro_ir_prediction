from dataclasses import dataclass, field

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
    early_stopping_patience: int = 500
    val_every_n_epochs: int = 5
    log_every_n_epochs: int = 5
    precision: str = "16-mixed"
    accelerator: str =  "gpu"
    devices: int = 1

@dataclass
class InferenceConfig:
    background_idx: int = 0
    bg_threshold: float = 0.5
    top_k_save: int = 3
    batch_size: int = 512
    devices: list = field(default_factory=lambda: [0, 1])
    pred_store_path: str = "/app/outputs/predictions_raw.lmdb"
    ckpt_path: str = ""


@dataclass
class DataConfig:
    zarr_path: str = ""
    spectra_per_class: int = 8
    batch_size: int = 64
    train_split_size: float = 0.5
    val_split_size: float = 0.5
    samples_min: float = 0.5
    background_max: float = 0.1

    mie_ratio: float = 0.0
    poly_ratio: float = 0.6
    bkg_poly_ratio: float = 0.6
    noise_ratio: float = 0.6
    co2_ratio: float = 0.2

    param_ranges: list[list[float]] = field(default_factory=list)
    bkg_param_ranges: list[list[float]] = field(default_factory=list)
    co2_params: list[float] = field(default_factory=list)

    max_noise_level: float = 0.05
    theta_min: float = 0.2
    theta_max: float = 0.45
    n0_min: float = 1.25
    n0_max: float = 1.65
    r_min: float = 2.0
    r_max: float = 14.0
    n_imag_min: float = 1e-4
    n_imag_max: float = 1e-2
    h_min: float = 1.5
    h_max: float = 2.5
    scale_min: float = 1.5
    scale_max: float = 2.5

@dataclass
class MasterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

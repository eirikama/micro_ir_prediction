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
    devices: int = 1


@dataclass
class DataConfig:
    zarr_path: str = ""
    spectra_per_plastic: int = 8
    batch_size: int = 64
    mie_ratio: float = 0.0
    poly_ratio: float = 0.6
    bkg_poly_ratio: float = 0.6
    noise_ratio: float = 0.6
    co2_ratio: float = 0.2

    param_ranges: list[list[float]] = field(default_factory=list)
    bkg_param_ranges: list[list[float]] = field(default_factory=list)
    co2_params: list[float] = field(default_factory=list)


@dataclass
class MasterConfig:
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig

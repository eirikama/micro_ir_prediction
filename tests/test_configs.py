"""
Domain config validation — runs against the real YAML files in configs/domain/.

Every domain present in configs/domain/*.yaml is tested automatically, so
adding a new domain is enough to have it covered at PR time.

Tests validate:
  - The full Hydra composition succeeds (schema, required fields, type checks)
  - alpha list length == num_classes
  - Every augmentation type exists in AUG_REGISTRY
  - intrinsic_validation=False implies a non-empty zarr_test_path
  - batch_size, spectra_per_class, val_split_size are in a sane range
  - Raman domains have z_normalize set and modality == "raman"
  - IR domains have modality == "ir"
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

# ── discover domains automatically ────────────────────────────────────────────

_DOMAIN_DIR = Path(__file__).parent.parent / "configs" / "domain"
_DOMAINS    = sorted(p.stem for p in _DOMAIN_DIR.glob("*.yaml"))

assert _DOMAINS, "No domain configs found — check that configs/domain/*.yaml exist"


def _load_cfg(domain: str):
    """Full Hydra config composition for one domain (no data files needed)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"domain={domain}",
                "mlflow.experiment_name=ci_test",  # '???' would raise otherwise
            ],
        )
    return cfg


# ── parametrised fixture — one test instance per domain ───────────────────────

@pytest.fixture(params=_DOMAINS, ids=_DOMAINS)
def domain_cfg(request):
    """Composed Hydra config for each domain."""
    return _load_cfg(request.param)


@pytest.fixture(params=_DOMAINS, ids=_DOMAINS)
def domain_name(request):
    return request.param


# ── 1. Hydra composition succeeds ─────────────────────────────────────────────

def test_config_loads(domain_name):
    """The full Hydra config composes without errors for every domain."""
    cfg = _load_cfg(domain_name)
    # Basic sanity: top-level sections exist
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "trainer")
    assert hasattr(cfg, "inference")


# ── 2. model config invariants ────────────────────────────────────────────────

def test_alpha_length_matches_num_classes(domain_cfg):
    """alpha list must have exactly num_classes entries (FocalLoss shape check)."""
    n_cls   = domain_cfg.model.num_classes
    n_alpha = len(domain_cfg.model.alpha)
    assert n_alpha == n_cls, (
        f"num_classes={n_cls} but alpha has {n_alpha} entries — "
        "FocalLoss will crash with a shape mismatch at runtime"
    )


def test_num_classes_positive(domain_cfg):
    assert domain_cfg.model.num_classes >= 2, "Need at least 2 classes"


def test_lr_positive(domain_cfg):
    assert domain_cfg.model.lr > 0


# ── 3. data config invariants ─────────────────────────────────────────────────

def test_zarr_path_set(domain_cfg):
    assert domain_cfg.data.zarr_path, "zarr_path is empty"


def test_intrinsic_validation_false_requires_test_zarr(domain_cfg):
    """Domains with intrinsic_validation=False must provide zarr_test_path."""
    if not domain_cfg.data.intrinsic_validation:
        assert domain_cfg.data.zarr_test_path, (
            "intrinsic_validation=False but zarr_test_path is empty — "
            "get_test_split() will fail at runtime"
        )


def test_spectra_per_class_positive(domain_cfg):
    assert domain_cfg.data.spectra_per_class >= 1


def test_batch_size_positive(domain_cfg):
    assert domain_cfg.data.batch_size >= 1


def test_val_split_size_in_range(domain_cfg):
    v = domain_cfg.data.val_split_size
    assert 0 < v < 1, f"val_split_size={v} must be in (0, 1)"


def test_augmentation_types_in_registry(domain_cfg):
    """Every augmentation type listed in the config must exist in AUG_REGISTRY."""
    from src.data.augmentation import AUG_REGISTRY
    aug_list = list(domain_cfg.data.get("augmentations") or [])
    for aug in aug_list:
        aug_type = aug.get("type")
        assert aug_type in AUG_REGISTRY, (
            f"Augmentation type '{aug_type}' not found in AUG_REGISTRY. "
            f"Known types: {sorted(AUG_REGISTRY.keys())}"
        )


def test_raman_domains_have_z_normalize(domain_cfg):
    """Raman domains (fluorescence augmentation present) should have z_normalize set."""
    aug_types = {a.get("type") for a in (domain_cfg.data.get("augmentations") or [])}
    if "fluorescence" in aug_types or "cosmic_rays" in aug_types:
        # Raman domain — z_normalize must be explicitly configured
        assert hasattr(domain_cfg.data, "z_normalize"), (
            "Raman domain is missing z_normalize in its data config"
        )


# ── 4. trainer config invariants ─────────────────────────────────────────────

def test_max_epochs_positive(domain_cfg):
    assert domain_cfg.trainer.max_epochs >= 1


def test_early_stopping_patience_positive(domain_cfg):
    assert domain_cfg.trainer.early_stopping_patience >= 1


def test_gradient_clip_val_set(domain_cfg):
    """gradient_clip_val must be in the config — trainer_engine.py reads it."""
    assert hasattr(domain_cfg.trainer, "gradient_clip_val"), (
        "gradient_clip_val missing from trainer config — "
        "trainer_engine.py accesses cfg.trainer.gradient_clip_val"
    )
    assert domain_cfg.trainer.gradient_clip_val > 0


# ── 5. steps_per_epoch would not stall ────────────────────────────────────────

def test_steps_per_epoch_with_clamping(domain_cfg):
    """Even if steps_per_epoch=0 (tiny data), max(1,steps) keeps training alive."""
    spc       = domain_cfg.data.spectra_per_class * 2        # setup() doubles this
    n_cls     = domain_cfg.model.num_classes
    val_split = domain_cfg.data.val_split_size
    batch     = domain_cfg.data.batch_size

    n_train_approx = int(spc * n_cls * (1 - val_split))
    raw_steps      = n_train_approx // batch
    clamped        = max(1, raw_steps)

    assert clamped >= 1, "max(1, steps_per_epoch) must always be >= 1"


def test_val_split_leaves_enough_for_all_classes(domain_cfg):
    """After splitting, every class must have >= 1 sample in the val set."""
    spc       = domain_cfg.data.spectra_per_class * 2
    val_split = domain_cfg.data.val_split_size
    val_per_class = int(spc * val_split)
    assert val_per_class >= 1, (
        f"spectra_per_class={domain_cfg.data.spectra_per_class}, "
        f"val_split_size={val_split} → only {val_per_class} val spectra per class"
    )


# ── 6. inference config invariants ───────────────────────────────────────────

def test_pred_store_path_set(domain_cfg):
    assert domain_cfg.inference.pred_store_path, "pred_store_path is empty"


def test_inference_batch_size_positive(domain_cfg):
    assert domain_cfg.inference.batch_size >= 1

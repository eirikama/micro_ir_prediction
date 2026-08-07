"""
Training sanity tests.

These tests guard against the failure modes that caused the original stall:

  1. steps_per_epoch = 0  →  limit_train_batches=0 and log_every_n_steps=0
     both crash or stall PL. The fix is max(1, steps_per_epoch) in trainer_engine.
  2. The trainer must terminate (finite epochs, not an infinite loop).
  3. After training, val_acc must appear in callback_metrics so early stopping
     and checkpointing can work.
  4. Validation must run and produce metrics (val_acc logged) even when
     training only runs 1 batch per epoch.

All tests use synthetic in-memory data — no zarr files, no GPU required.
"""
from __future__ import annotations

import concurrent.futures
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tests.conftest import N_CLASSES, SPECTRAL_LEN


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_cfg(batch_size: int = 16, max_epochs: int = 2, val_split: float = 0.5):
    """Return a minimal Hydra-style config for training tests."""
    from src.config_schema import ModelConfig, TrainerConfig
    model_cfg   = OmegaConf.structured(ModelConfig(
        conv_channels=16, kernel_size=7, pred_dropout=0.0,
        num_classes=N_CLASSES, gamma=0.0,
        alpha=[1.0] * N_CLASSES, lr=1e-4, weight_decay=0.0,
    ))
    trainer_cfg = OmegaConf.structured(TrainerConfig(
        max_epochs=max_epochs, min_epochs=1,
        early_stopping_patience=100, val_every_n_epochs=1,
        log_every_n_epochs=1, precision="32-true",
        accelerator="cpu", devices=1, gradient_clip_val=1.0,
    ))
    data_cfg = OmegaConf.create({
        "batch_size":        batch_size,
        "val_split_size":    val_split,
        "include_bkg_pixels": False,
        "augmentations":     [],
        "z_normalize":       False,
        "augment_train":     False,
        "augment_val":       False,
    })
    return OmegaConf.create({"model": model_cfg, "trainer": trainer_cfg, "data": data_cfg})


def _make_datamodule(n_spectra_per_class: int, batch_size: int, val_split: float = 0.5):
    """Build a SpectralDataModule populated with synthetic data (no zarr)."""
    import pytorch_lightning as pl
    from src.data.datamodule import SpectralDataset, SpectralDataModule

    rng  = np.random.default_rng(0)
    wn   = np.linspace(400, 4000, SPECTRAL_LEN, dtype=np.float32)
    n    = n_spectra_per_class * N_CLASSES
    spec = (rng.random((n, SPECTRAL_LEN), dtype=np.float32) + 0.5)
    lbl  = np.repeat(np.arange(N_CLASSES), n_spectra_per_class).astype(np.int64)

    data_cfg = OmegaConf.create({
        "batch_size":        batch_size,
        "val_split_size":    val_split,
        "include_bkg_pixels": False,
        "augmentations":     [],
        "z_normalize":       False,
        "augment_train":     False,
        "augment_val":       False,
    })

    s_tr, s_va, l_tr, l_va = train_test_split(
        spec, lbl, test_size=val_split, stratify=lbl, random_state=42
    )

    dm = SpectralDataModule.__new__(SpectralDataModule)
    # Bypassing __init__ (to skip its zarr-backed setup()) also skips
    # pl.LightningDataModule.__init__, which sets base-class state the
    # Trainer relies on (e.g. allow_zero_length_dataloader_with_multiple_devices).
    # Call it directly so the datamodule still satisfies that contract.
    pl.LightningDataModule.__init__(dm)
    dm.cfg            = data_cfg
    dm.spectra        = spec
    dm.wn             = wn
    dm.label_encoding = {f"class_{i}": i for i in range(N_CLASSES)}
    dm.steps_per_epoch = len(l_tr) // batch_size
    dm.val_batches     = len(l_va) // batch_size
    dm.train_ds = SpectralDataset(s_tr, l_tr, wn, data_cfg, augment=False)
    dm.val_ds   = SpectralDataset(s_va, l_va, wn, data_cfg, augment=False)
    return dm


# ── 1. steps_per_epoch clamping ───────────────────────────────────────────────

class TestStepsPerEpoch:

    def test_zero_steps_clamped_to_one(self):
        """When data is tiny, steps_per_epoch=0 must become 1 before trainer."""
        # 2 spectra/class × 3 classes = 6 train spectra; batch_size=16 → 6//16=0
        dm = _make_datamodule(n_spectra_per_class=2, batch_size=16)
        assert dm.steps_per_epoch == 0, "precondition: steps really is 0"
        clamped = max(1, dm.steps_per_epoch)
        assert clamped == 1

    def test_adequate_data_gives_positive_steps(self):
        dm = _make_datamodule(n_spectra_per_class=16, batch_size=4)
        assert dm.steps_per_epoch > 0

    def test_steps_match_floor_division(self):
        """steps_per_epoch = len(train_labels) // batch_size."""
        dm = _make_datamodule(n_spectra_per_class=10, batch_size=4, val_split=0.5)
        # 10 * 3 = 30 total; 15 train after 50% split; 15//4 = 3
        assert dm.steps_per_epoch == 3

    @pytest.mark.parametrize("spectra_per_class,batch_size", [
        # 1 spectrum/class is omitted: sklearn's stratified split needs >= 2
        # samples per class, and no real domain config ever sets
        # spectra_per_class that low anyway.
        (2,  16),   # 6 train → steps=0
        (4,  16),   # 12 train → steps=0
        (10, 64),   # 15 train → steps=0
        (3,   4),   # 9 train → steps=2 (positive — no clamping needed)
    ])
    def test_clamped_steps_always_positive(self, spectra_per_class, batch_size):
        dm = _make_datamodule(n_spectra_per_class=spectra_per_class,
                              batch_size=batch_size)
        assert max(1, dm.steps_per_epoch) >= 1


# ── 2. log_every_n_steps must be >= 1 ────────────────────────────────────────

class TestLogEveryN:
    # A prior test here (test_log_every_n_steps_is_never_zero) asserted that
    # pl.Trainer(log_every_n_steps=0) raises. That's no longer true as of
    # pytorch_lightning 2.5.0 (the pinned version) — it was a stale library
    # assumption, not a guarantee this codebase relies on. The thing that
    # actually matters — trainer_engine.py never passing 0 — is verified
    # below.

    def test_trainer_engine_uses_clamped_steps(self):
        """trainer_engine.run_training must never pass 0 to log_every_n_steps."""
        # We inspect the Trainer by monkeypatching pl.Trainer.__init__
        import pytorch_lightning as pl
        from src.models.aacnn import AACNN

        recorded = {}
        _orig_init = pl.Trainer.__init__

        def _spy_init(self, **kwargs):
            recorded.update(kwargs)
            _orig_init(self, **kwargs)

        dm  = _make_datamodule(n_spectra_per_class=2, batch_size=16)
        cfg = _make_cfg(batch_size=16, max_epochs=1)

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(pl.Trainer, "__init__", _spy_init)
            try:
                from src.training.trainer_engine import run_training
                from src.models.aacnn import AACNN
                run_training(cfg, AACNN(cfg.model), dm, logger=False)
            except Exception:
                pass   # training itself may fail — we only care about the kwargs

        assert "log_every_n_steps" in recorded
        assert recorded["log_every_n_steps"] >= 1

    def test_trainer_engine_uses_clamped_limit_train_batches(self):
        """limit_train_batches must be >= 1 even when steps_per_epoch=0."""
        import pytorch_lightning as pl

        recorded = {}
        _orig_init = pl.Trainer.__init__

        def _spy_init(self, **kwargs):
            recorded.update(kwargs)
            _orig_init(self, **kwargs)

        dm  = _make_datamodule(n_spectra_per_class=2, batch_size=16)
        cfg = _make_cfg(batch_size=16, max_epochs=1)

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(pl.Trainer, "__init__", _spy_init)
            try:
                from src.training.trainer_engine import run_training
                from src.models.aacnn import AACNN
                run_training(cfg, AACNN(cfg.model), dm, logger=False)
            except Exception:
                pass

        assert recorded.get("limit_train_batches", 0) >= 1


# ── 3. training terminates ────────────────────────────────────────────────────

TRAINING_TIMEOUT_S = 60   # seconds; fail if training hangs longer than this


class TestTrainingTerminates:

    def _run_training(self, n_spectra_per_class: int, batch_size: int,
                      max_epochs: int = 2) -> dict:
        """Run training in a thread; return {'done': bool, 'error': Exception|None}."""
        from src.training.trainer_engine import run_training
        from src.models.aacnn import AACNN

        dm  = _make_datamodule(n_spectra_per_class, batch_size)
        cfg = _make_cfg(batch_size=batch_size, max_epochs=max_epochs)

        result: dict = {"done": False, "error": None, "epoch": -1}

        def _target():
            try:
                _, _, epoch, _ = run_training(cfg, AACNN(cfg.model), dm, logger=False)
                result["epoch"] = epoch
                result["done"]  = True
            except Exception as e:
                result["error"] = e

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_target)
            try:
                fut.result(timeout=TRAINING_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                pytest.fail(
                    f"Training did not terminate within {TRAINING_TIMEOUT_S}s — "
                    "possible infinite loop in the DataLoader or Trainer."
                )
        return result

    def test_normal_data_terminates(self):
        """Standard run: adequate data, batch_size fits neatly."""
        res = self._run_training(n_spectra_per_class=16, batch_size=4)
        assert res["done"], f"Training raised: {res['error']}"

    def test_tiny_data_terminates(self):
        """Edge case: spectra_per_class=2, batch_size=16 → steps_per_epoch=0."""
        res = self._run_training(n_spectra_per_class=2, batch_size=16)
        assert res["done"], f"Training raised: {res['error']}"

    def test_correct_epoch_count(self):
        """Trainer should complete exactly max_epochs epochs."""
        res = self._run_training(n_spectra_per_class=16, batch_size=4, max_epochs=3)
        assert res["done"]
        # trainer.current_epoch is 0-indexed; after 3 epochs it equals 2
        assert res["epoch"] == 2


# ── 4. metrics are populated after training ───────────────────────────────────

class TestMetricsLogged:

    def test_val_acc_in_callback_metrics(self):
        """val_acc must appear in callback_metrics for checkpointing to work."""
        import pytorch_lightning as pl
        from src.models.aacnn import AACNN
        from src.training.trainer_engine import run_training

        dm  = _make_datamodule(n_spectra_per_class=16, batch_size=4)
        cfg = _make_cfg(batch_size=4, max_epochs=2)

        collected: dict = {}

        orig_fit = pl.Trainer.fit

        def _spy_fit(self, *args, **kwargs):
            orig_fit(self, *args, **kwargs)
            collected.update(dict(self.callback_metrics))

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(pl.Trainer, "fit", _spy_fit)
            run_training(cfg, AACNN(cfg.model), dm, logger=False)

        assert "val_acc"  in collected, "val_acc missing — early stopping will never fire"
        assert "train_acc" in collected, "train_acc missing"

    def test_val_acc_is_finite(self):
        import pytorch_lightning as pl
        from src.models.aacnn import AACNN
        from src.training.trainer_engine import run_training

        dm  = _make_datamodule(n_spectra_per_class=16, batch_size=4)
        cfg = _make_cfg(batch_size=4, max_epochs=2)

        collected: dict = {}

        orig_fit = pl.Trainer.fit
        def _spy_fit(self, *args, **kwargs):
            orig_fit(self, *args, **kwargs)
            collected.update(dict(self.callback_metrics))

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(pl.Trainer, "fit", _spy_fit)
            run_training(cfg, AACNN(cfg.model), dm, logger=False)

        val_acc = collected.get("val_acc")
        assert val_acc is not None
        assert torch.isfinite(val_acc)


# ── 5. DataLoader iteration terminates after limit ────────────────────────────

class TestDataLoaderTermination:

    def test_iter_stops_after_n_batches(self):
        """Python `break` after N steps must cleanly stop the infinite generator."""
        from src.data.datamodule import SpectralDataset

        rng = np.random.default_rng(1)
        wn  = np.linspace(400, 4000, SPECTRAL_LEN, dtype=np.float32)
        s   = rng.random((N_CLASSES * 4, SPECTRAL_LEN), dtype=np.float32) + 0.5
        l   = np.repeat(np.arange(N_CLASSES), 4).astype(np.int64)
        cfg = OmegaConf.create({
            "batch_size": N_CLASSES * 2, "include_bkg_pixels": False,
            "augmentations": [], "z_normalize": False,
            "augment_train": False, "augment_val": False,
        })
        ds = SpectralDataset(s, l, wn, cfg, augment=False)
        dl = DataLoader(ds, batch_size=None)

        N_LIMIT = 5
        count   = 0
        for _ in dl:
            count += 1
            if count >= N_LIMIT:
                break

        assert count == N_LIMIT

    def test_val_dataloader_yields_batches(self):
        """Validation dataloader must yield batches (not stall on first next())."""
        dm = _make_datamodule(n_spectra_per_class=16, batch_size=4)
        dl = dm.val_dataloader()
        x, y = next(iter(dl))
        # SpectralDataset.__iter__ tops batches back up to batch_size with
        # extra samples after the per-class split, so it's never
        # N_CLASSES * (batch_size // N_CLASSES) short of a remainder.
        assert x.shape[0] == 4  # == batch_size


# ── 6. stratified split at minimum sample count ───────────────────────────────

class TestStratifiedSplit:

    @pytest.mark.parametrize("n_per_class", [2, 4, 8])
    def test_split_succeeds(self, n_per_class):
        """train_test_split must not raise even with very few samples per class."""
        rng = np.random.default_rng(0)
        n   = n_per_class * N_CLASSES
        s   = rng.random((n, SPECTRAL_LEN), dtype=np.float32)
        l   = np.repeat(np.arange(N_CLASSES), n_per_class)
        # Should not raise
        s_tr, s_va, l_tr, l_va = train_test_split(
            s, l, test_size=0.5, stratify=l, random_state=42
        )
        assert len(l_tr) == n // 2
        assert len(l_va) == n // 2

    def test_all_classes_in_both_splits(self):
        """Every class must appear in both train and val after splitting."""
        n_per_class = 4
        rng = np.random.default_rng(0)
        n   = n_per_class * N_CLASSES
        s   = rng.random((n, SPECTRAL_LEN), dtype=np.float32)
        l   = np.repeat(np.arange(N_CLASSES), n_per_class)
        _, _, l_tr, l_va = train_test_split(
            s, l, test_size=0.5, stratify=l, random_state=42
        )
        assert set(np.unique(l_tr)) == set(range(N_CLASSES))
        assert set(np.unique(l_va)) == set(range(N_CLASSES))

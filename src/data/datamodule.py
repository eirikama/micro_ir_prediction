from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset

# Restore removed aliases for legacy Cython extensions (removed in NumPy 1.24)
for _alias, _target in [
    ("complex", np.complex128),
    ("float",   np.float64),
    ("int",     np.int_),
    ("bool",    np.bool_),
]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

from src.data.augmentation import AUG_REGISTRY, _apply_fluorescence
from src.data.preprocessing import apply_preprocessing, fit_preprocessing, resolve_steps
from src.data.sampling import create_experiment_split, get_training_data_spectra, get_training_data_hyperspectral

log = logging.getLogger(__name__)


class SpectralDataset(IterableDataset):
    def __init__(
        self,
        spectra: np.ndarray,
        y: np.ndarray,
        wn: np.ndarray,
        cfg: DictConfig,
        augment: bool,
        preproc_state: dict | None = None,
    ) -> None:

        self.spectra = spectra.cpu().numpy() if torch.is_tensor(spectra) else spectra
        self.y = y.cpu().numpy() if torch.is_tensor(y) else y
        self.wn = wn.cpu().numpy() if torch.is_tensor(wn) else wn

        self.cfg = cfg
        self.augment = augment
        # Fitted state for stateful preprocessing steps (e.g. the EMSC
        # reference). Fitted on the training split in SpectralDataModule.setup
        # and shared with the val dataset, so val is corrected against the
        # training reference rather than its own.
        self.preproc_state = preproc_state or {}

        # 1. Group indices by class
        self.unique_classes = np.unique(self.y)
        self.num_classes = len(self.unique_classes)
        self.class_indices = {c: np.where(self.y == c)[0] for c in self.unique_classes}

        # 2. Determine how many samples per class per batch
        self.samples_per_class = self.cfg.batch_size // self.num_classes
        self.remainder = self.cfg.batch_size % self.num_classes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            batch_idx = []

            # 3. Pull an equal number of samples from each class
            for c in self.unique_classes:
                indices = self.class_indices[c]
                # 'replace=True' allows batch_size > dataset size
                chosen = np.random.choice(indices, self.samples_per_class, replace=True)
                batch_idx.extend(chosen)

            # 4. Handle remainder if batch_size is not divisible by num_classes
            if self.remainder > 0:
                extra_indices = np.random.choice(
                    np.arange(len(self.y)), self.remainder, replace=True
                )
                batch_idx.extend(extra_indices)

            # 5. Shuffle the batch so the model doesn't see classes in order
            batch_idx = np.array(batch_idx)
            np.random.shuffle(batch_idx)

            yield self._process_batch(batch_idx)

    def _process_batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.spectra[idx].copy()
        y = self.y[idx]
        B = s.shape[0]

        if self.augment:
            use_background = getattr(self.cfg, "include_bkg_pixels", False)
            if use_background:
                is_signal = y != 0
            else:
                is_signal = np.ones(B, dtype=bool)

            aug_map = self.cfg.get("augmentations") or {}
            for aug_name, aug_cfg in aug_map.items():
                if not aug_cfg.get("enabled", True):
                    continue

                aug_type = aug_cfg.get("type", aug_name)

                if use_background and aug_cfg.get("signal_only", False):
                    base = is_signal
                elif use_background and aug_cfg.get("background_only", False):
                    base = ~is_signal
                else:
                    base = np.ones(B, dtype=bool)

                mask = base & (np.random.rand(B) < aug_cfg.ratio)
                if mask.any():
                    s = AUG_REGISTRY[aug_type](s, mask, self.wn, aug_cfg)

        # Preprocessing runs AFTER augmentation and on train, val and
        # inference alike: augmentation simulates instrument artifacts,
        # preprocessing removes them. Independent of `augment`, so it can be
        # used instead of augmentation, alongside it, or not at all.
        s, _ = apply_preprocessing(
            s, self.wn, self.cfg.get("preprocessing"), self.preproc_state
        )

        if self.cfg.z_normalize:
            mu = s.mean(axis=1, keepdims=True)
            sigma = s.std(axis=1, keepdims=True)
            s = (s - mu) / np.maximum(sigma, 1e-3)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(s).float().unsqueeze(1), torch.from_numpy(y).long()


class SpectralDataModule(pl.LightningDataModule):
    def __init__(self, split: list[dict[str, str]], cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.train_ds = None
        self.val_ds = None
        self.steps_per_epoch = 0

    def _sample_training_data(
        self,
        spectra_per_class: int,
        bkg_per_class: int,
        seed: int = 42,
    ):
        """Draw spectra from this run's training split.

        Factored out of setup() so the fluorescence fit pool can be drawn
        through exactly the same path (and from the same split) as the
        training data itself, just with a different budget and seed.
        """
        if self.cfg.hyperspectra:
            return get_training_data_hyperspectral(
                split=self.split,
                zarr_path=self.cfg.zarr_path,
                spectra_per_class=spectra_per_class,
                bkg_per_class=bkg_per_class,
                patch_size=self.cfg.sampling_patch_size,
                background_max=self.cfg.background_max,
                sample_min=self.cfg.sample_min,
                max_class_attempts=self.cfg.max_sampling_per_class_attempts,
                include_bkg_pixels=self.cfg.include_bkg_pixels,
                seed=seed,
            )
        return get_training_data_spectra(
            split=self.split,
            zarr_path=self.cfg.zarr_path,
            spectra_per_class=spectra_per_class,
            classes=list(self.cfg.classes) if self.cfg.get("classes") else None,
            seed=seed,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is not None:
            return

        bkg_per_class = self.cfg.spectra_per_class // self.cfg.sample_to_bkg_spectra_ratio
        label, spectra, wn, label_encoding = self._sample_training_data(
            spectra_per_class=self.cfg.spectra_per_class * 2,
            bkg_per_class=bkg_per_class * 2,
        )
        self.wn = wn
        self.spectra = spectra
        self.label_encoding = label_encoding

        s_train, s_val, l_train, l_val = train_test_split(
            spectra, label, test_size=self.cfg.val_split_size, stratify=label, random_state=42
        )

        self.steps_per_epoch = len(l_train) // self.cfg.batch_size
        self.val_batches = len(l_val) // self.cfg.batch_size

        # Fit stateful preprocessing (e.g. the EMSC reference) on the TRAIN
        # split only, then share that state with val — otherwise each split
        # is corrected against its own reference and they are not comparable.
        # The same state is written beside the checkpoint after training so
        # inference reproduces it exactly (see save_preproc_state).
        steps = resolve_steps(self.cfg.get("preprocessing"))
        self.preproc_state = fit_preprocessing(s_train, self.wn, self.cfg.get("preprocessing"))
        if steps:
            log.info(
                "[preprocessing] enabled: %s", " -> ".join(s["type"] for s in steps)
            )
        else:
            log.info("[preprocessing] disabled")

        self.train_ds = SpectralDataset(s_train, l_train, self.wn, self.cfg,
                                        self.cfg.augment_train, self.preproc_state)
        self.val_ds = SpectralDataset(s_val, l_val, self.wn, self.cfg,
                                      self.cfg.augment_train and self.cfg.augment_val,
                                      self.preproc_state)

        # Fit fluorescence PCA basis when that augmentation is enabled.
        if self.cfg.augment_train or self.cfg.augment_val:
            aug_map = self.cfg.get("augmentations") or {}
            for aug_name, aug_cfg_item in aug_map.items():
                aug_type = aug_cfg_item.get("type", aug_name)
                if aug_type == "fluorescence" and aug_cfg_item.get("enabled", True):
                    self._fit_fluorescence_basis(aug_cfg_item, s_train, len(label_encoding))
                    break

    def _fit_fluorescence_basis(self, aug_cfg, s_train: np.ndarray, n_classes: int) -> None:
        """Fit the fluorescence PCA basis, optionally on a fixed reference pool.

        Two things this controls, both of which otherwise bias a
        scaling-law experiment (accuracy vs spectra_per_class):

        1. The basis is fitted on the TRAIN split only. It used to be fitted
           on `self.spectra`, i.e. before the train/val split — with
           val_split_size 0.5 that put half the validation set into the fit.
           Only background shape leaks (not labels), but val_acc drives early
           stopping and checkpoint selection, so it biased model selection.

        2. `fit_pool_size` (K) draws a dedicated, N-independent pool of K
           spectra for the fit instead of using whatever this run happens to
           have sampled. Without it the basis is fitted on
           spectra_per_class * 2 * n_classes spectra, so the augmentation is
           a *different transformation* at every point of the scaling curve —
           the augmentation being measured changes along the x-axis. Drawing
           K from the same training split at every N decouples the two.

           Set fit_pool_size to null/0 to switch this off and fit on the
           run's own training spectra (the N-coupled behaviour), which is
           what you want as the "B" arm of an A/B on this effect.

        Config keys
        -----------
        fit_pool_size : int | null — K, spectra drawn for the fit. null/0 =>
                        use this run's training spectra.
        fit_pool_seed : int | null — seed for the pool draw. Fixed by default
                        so the pool (and hence the basis) is identical across
                        runs and across N. null => follow cfg.seed, which
                        makes the pool vary per seed like the rest of the data.
        """
        pool_size = aug_cfg.get("fit_pool_size", None)
        pool_size = int(pool_size) if pool_size else 0

        if pool_size <= 0:
            log.info(
                "[fluorescence] fit_pool_size unset — fitting on this run's "
                "%d training spectra (basis is coupled to spectra_per_class)",
                s_train.shape[0],
            )
            _apply_fluorescence.fit(self.wn, s_train, aug_cfg)
            log.info("[fluorescence] fit complete")
            return

        pool_seed = aug_cfg.get("fit_pool_seed", 12345)
        pool_seed = int(self.cfg.get("seed", 42)) if pool_seed is None else int(pool_seed)

        # Budget per class, rounded up so the pool reaches K even when K does
        # not divide evenly. The draw is capped by what the split actually
        # holds, so a small split still yields a smaller pool — K fixes the
        # *request*, not the available support.
        per_class = -(-pool_size // max(n_classes, 1))
        bkg_per_class = per_class // max(self.cfg.sample_to_bkg_spectra_ratio, 1)

        log.info(
            "[fluorescence] drawing a fixed fit pool: K=%d (%d/class, seed=%d)",
            pool_size, per_class, pool_seed,
        )
        _, pool_spectra, _, _ = self._sample_training_data(
            spectra_per_class=per_class,
            bkg_per_class=bkg_per_class,
            seed=pool_seed,
        )

        if pool_spectra.shape[0] > pool_size:
            idx = np.random.default_rng(pool_seed).choice(
                pool_spectra.shape[0], pool_size, replace=False
            )
            pool_spectra = pool_spectra[idx]

        log.info(
            "[fluorescence] fitting on fixed pool of %d spectra "
            "(independent of spectra_per_class=%d)",
            pool_spectra.shape[0], self.cfg.spectra_per_class,
        )
        _apply_fluorescence.fit(self.wn, pool_spectra, aug_cfg)
        log.info("[fluorescence] fit complete")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=None, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=None, pin_memory=False)

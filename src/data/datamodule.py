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
from src.data.sampling import create_experiment_split, get_training_data

log = logging.getLogger(__name__)


class SpectralDataset(IterableDataset):
    def __init__(self, spectra: np.ndarray, y: np.ndarray, wn: np.ndarray, cfg: DictConfig, augment: bool) -> None:

        self.spectra = spectra.cpu().numpy() if torch.is_tensor(spectra) else spectra
        self.y = y.cpu().numpy() if torch.is_tensor(y) else y
        self.wn = wn.cpu().numpy() if torch.is_tensor(wn) else wn

        self.cfg = cfg
        self.augment = augment

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

    def setup(self, stage: str | None = None) -> None:

        bkg_per_class = self.cfg.spectra_per_class // self.cfg.sample_to_bkg_spectra_ratio
        label, spectra, wn, label_encoding = get_training_data(
            split=self.split,
            zarr_path=self.cfg.zarr_path,
            spectra_per_class=self.cfg.spectra_per_class * 2,
            bkg_per_class=bkg_per_class * 2,
            patch_size=self.cfg.sampling_patch_size,
            background_max=self.cfg.background_max,
            sample_min=self.cfg.sample_min,
            max_class_attempts=self.cfg.max_sampling_per_class_attempts,
            include_bkg_pixels=self.cfg.include_bkg_pixels
        )
        self.wn = wn
        self.spectra = spectra
        self.label_encoding = label_encoding

        s_train, s_val, l_train, l_val = train_test_split(
            spectra, label, test_size=self.cfg.val_split_size, stratify=label, random_state=42
        )

        self.steps_per_epoch = len(l_train) // self.cfg.batch_size
        self.val_batches = len(l_val) // self.cfg.batch_size

        self.train_ds = SpectralDataset(s_train, l_train, self.wn, self.cfg, self.cfg.augment_train)
        self.val_ds = SpectralDataset(s_val, l_val, self.wn, self.cfg, self.cfg.augment_train and self.cfg.augment_val)

        # Fit fluorescence PCA basis when that augmentation is enabled.
        if self.cfg.augment_train or self.cfg.augment_val:
            aug_map = self.cfg.get("augmentations") or {}
            for aug_name, aug_cfg_item in aug_map.items():
                aug_type = aug_cfg_item.get("type", aug_name)
                if aug_type == "fluorescence" and aug_cfg_item.get("enabled", True):
                    log.info("[fluorescence] fitting on %d spectra", self.spectra.shape[0])
                    _apply_fluorescence.fit(self.wn, self.spectra, aug_cfg_item)
                    log.info("[fluorescence] fit complete")
                    break

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=None, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=None, pin_memory=False)

from typing import Iterator
import numpy as np
import zarr
import random
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig

import numpy as np
# Restore removed aliases for legacy Cython extensions (removed in NumPy 1.24)
for _alias, _target in [
    ("complex", np.complex128),
    ("float",   np.float64),
    ("int",     np.int_),
    ("bool",    np.bool_),
]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

from src.data.augmentation import add_co2, add_polynomial, add_scattering #, add_cylindrical_scattering
from src.data.sampling import create_experiment_split, get_training_data


class SpectralDataset(IterableDataset):
    def __init__(self, spectra: np.ndarray, y: np.ndarray, wn: np.ndarray, cfg: DictConfig) -> None:

        self.spectra = spectra.cpu().numpy() if torch.is_tensor(spectra) else spectra
        self.y = y.cpu().numpy() if torch.is_tensor(y) else y
        self.wn = wn.cpu().numpy() if torch.is_tensor(wn) else wn

        self.cfg = cfg

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
        is_signal = y != 0
        # is_signal = y < 10

        # --- Mie Augmentation ---
        mie_mask = is_signal & (np.random.rand(B) < self.cfg.mie_ratio)
        if mie_mask.any():
            s_subset = s[mie_mask]
            s_subset -= s_subset.min(axis=1, keepdims=True)
            s_subset /= s_subset.max(axis=1, keepdims=True) + 1e-9

            n0s, rs, n_ims, hs, scs = (
                np.random.uniform(low, high, (s_subset.shape[0], 1))
                for low, high in [
                    (self.cfg.n0_min, self.cfg.n0_max),
                    (self.cfg.r_min, self.cfg.r_max),
                    (self.cfg.n_imag_min, self.cfg.n_imag_max),
                    (self.cfg.h_min, self.cfg.h_max),
                    (self.cfg.scale_min, self.cfg.scale_max),
                ]
            )
            theta = np.random.uniform(self.cfg.theta_min, self.cfg.theta_max)
            if True:
                s[mie_mask] = add_scattering(s_subset, self.wn, rs, n0s, n_ims, theta, hs, scs)
            else:
                s[mie_mask] = add_cylindrical_scattering(s_subset, self.wn, rs, n0s, n_ims, theta, hs, scs)

        # --- Vectorized Polynomials ---
        poly_mask = is_signal & (np.random.rand(B) < self.cfg.poly_ratio)
        bkg_mask = ~is_signal & (np.random.rand(B) < self.cfg.bkg_poly_ratio)

        for mask, ranges in zip(
            [poly_mask, bkg_mask],
            [self.cfg.param_ranges, self.cfg.bkg_param_ranges],
            strict=True
        ):
            if mask.any():
                n_samples = np.sum(mask)
                lows = np.array([r[0] for r in ranges])
                highs = np.array([r[1] for r in ranges])
                params = (np.random.rand(n_samples, 4) * (highs - lows) + lows).T
                s[mask] = add_polynomial(s[mask], self.wn, params)

        noise_mask = np.random.rand(B) < self.cfg.noise_ratio

        if noise_mask.any():
            n_noisy = noise_mask.sum()
            noise_scales = np.random.rand(n_noisy, 1) * self.cfg.max_noise_level
            s[noise_mask] += np.random.normal(0, 1, s[noise_mask].shape) * noise_scales

        co2_mask = np.random.rand(B) < self.cfg.co2_ratio

        if co2_mask.any():
            s_subset = s[co2_mask]
            s[co2_mask] = add_co2(s_subset, self.wn, self.cfg.co2_params)

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
        self.label_encoding = label_encoding

        s_train, s_val, l_train, l_val = train_test_split(
            spectra, label, test_size=self.cfg.val_split_size, stratify=label, random_state=42
        )

        self.steps_per_epoch = len(l_train) // self.cfg.batch_size
        self.val_batches = len(l_val) // self.cfg.batch_size

        self.train_ds = SpectralDataset(s_train, l_train, self.wn, self.cfg)
        self.val_ds = SpectralDataset(s_val, l_val, self.wn, self.cfg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=None, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=None, pin_memory=True)

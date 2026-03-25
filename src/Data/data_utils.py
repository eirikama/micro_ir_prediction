import random

import numpy as np
import pytorch_lightning as L
import torch
import zarr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset

from src.Data.aug_utils import add_co2, add_polynomial, add_scattering


def create_experiment_split(zarr_path, split_ratio=0.5):
    store = zarr.open(zarr_path, mode="r")
    images_group = store["images"]

    data_list = []
    for name in images_group.keys():
        label = images_group[name].attrs.get("label", "unknown")
        data_list.append({"name": name, "label": label})

    random.shuffle(data_list)
    mid = int(len(data_list) * split_ratio)
    split = {"train": data_list[:mid], "test": data_list[mid:]}
    return split


def get_training_data(
    split,
    zarr_path="/mnt/ssd3/eirik/ProcessedData/microplastics_library.zarr",
    N=8,
    patch_size=128,
):
    root = zarr.open(zarr_path, mode="r")
    images_group = root["images"]
    wn = np.asarray(root.attrs["wavenumbers"], dtype=np.float32)

    unique_labels = sorted(list(set(item["label"] for item in split)))
    rng = np.random.default_rng()

    spec_per_image = N // 8
    all_spectra, all_labels = [], []
    label_encoding = {"bkg": 0}
    for i, label in enumerate(unique_labels):
        names = [item["name"] for item in split if item["label"] == label]

        p_sample, p_bkg = [], []
        while len(p_sample) < N:
            name = names[rng.integers(len(names))]
            z_arr = images_group[name]["data"]

            H, W, L = z_arr.shape
            max_h = max(1, H - patch_size)
            max_w = max(1, W - patch_size)
            y0 = rng.integers(0, max_h) if H > patch_size else 0
            x0 = rng.integers(0, max_w) if W > patch_size else 0

            patch = z_arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
            spectra = patch.reshape(-1, L)
            means = spectra.mean(axis=1)

            spec_mask = means > 0.5
            bkg_mask = means < 0.1

            if spec_mask.any() and bkg_mask.any():
                valid = spectra[spec_mask]
                k = min(spec_per_image, len(valid))
                chosen = rng.choice(len(valid), size=k, replace=False)
                p_sample.extend(valid[chosen])

                valid = spectra[bkg_mask]
                k = min(spec_per_image, len(valid))
                chosen = rng.choice(len(valid), size=k, replace=False)
                p_bkg.extend(valid[chosen])

        N_bkg_per_plastic = int(len(p_sample) / 4)

        all_spectra.append(np.stack(p_sample[:N]))
        all_spectra.append(np.stack(p_bkg[:N_bkg_per_plastic]))

        all_labels.append(np.ones(N) * (i + 1))
        all_labels.append(np.zeros(N_bkg_per_plastic))
        label_encoding[label] = i + 1

    return np.hstack(all_labels), np.vstack(all_spectra), wn, label_encoding


class SpectralDataset(IterableDataset):
    def __init__(self, spectra, y, wn, config):
        self.spectra = spectra.cpu().numpy() if torch.is_tensor(spectra) else spectra
        self.y = y.cpu().numpy() if torch.is_tensor(y) else y
        self.wn = wn.cpu().numpy() if torch.is_tensor(wn) else wn

        self.config = config

        # 1. Group indices by class
        self.unique_classes = np.unique(self.y)
        self.num_classes = len(self.unique_classes)
        self.class_indices = {c: np.where(self.y == c)[0] for c in self.unique_classes}

        # 2. Determine how many samples per class per batch
        self.samples_per_class = self.config.batch_size // self.num_classes
        self.remainder = self.config.batch_size % self.num_classes

    def __iter__(self):
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

    def _process_batch(self, idx):
        s = self.spectra[idx].copy()
        y = self.y[idx]
        B = s.shape[0]
        is_signal = y != 0

        # --- Mie Augmentation ---
        mie_mask = is_signal & (np.random.rand(B) < self.config.mie_ratio)
        if mie_mask.any():
            s_subset = s[mie_mask]
            s_subset -= s_subset.min(axis=1, keepdims=True)
            s_subset /= s_subset.max(axis=1, keepdims=True) + 1e-9

            n0s, rs, n_ims, hs, scs = (
                np.random.uniform(low, high, (s_subset.shape[0], 1))
                for low, high in [
                    (1.25, 1.65),
                    (2.0, 14),
                    (1e-4, 1e-2),
                    (1.5, 2.5),
                    (1.5, 2.5),
                ]
            )
            theta_max = np.random.uniform(0.2, 0.45)
            s[mie_mask] = add_scattering(s_subset, self.wn, rs, n0s, n_ims, theta_max, hs, scs)

        # --- Vectorized Polynomials ---
        poly_mask = is_signal & (np.random.rand(B) < self.config.poly_ratio)
        bkg_mask = ~is_signal & (np.random.rand(B) < self.config.bkg_poly_ratio)

        for mask, ranges in zip(
            [poly_mask, bkg_mask],
            [self.config.param_ranges, self.config.bkg_param_ranges],
            strict=False,
        ):
            if mask.any():
                n_samples = np.sum(mask)
                lows = np.array([r[0] for r in ranges])
                highs = np.array([r[1] for r in ranges])
                params = (np.random.rand(n_samples, 4) * (highs - lows) + lows).T
                s[mask] = add_polynomial(s[mask], self.wn, params)

        noise_mask = np.random.rand(B) < self.config.noise_ratio

        if noise_mask.any():
            n_noisy = noise_mask.sum()
            noise_scales = np.random.rand(n_noisy, 1) * 0.05
            s[noise_mask] += np.random.normal(0, 1, s[noise_mask].shape) * noise_scales

        co2_mask = np.random.rand(B) < self.config.co2_ratio

        if co2_mask.any():
            s_subset = s[co2_mask]
            s[co2_mask] = add_co2(s_subset, self.wn, self.config.co2_params)

        return torch.from_numpy(s).float().unsqueeze(1), torch.from_numpy(y).long()


class SpectralDataModule(L.LightningDataModule):
    def __init__(self, split, config):
        super().__init__()
        self.config = config
        self.split = split

        self.train_ds = None
        self.val_ds = None
        self.steps_per_epoch = 0

    def setup(self, stage=None):
        label, spectra, wn, label_encoding = get_training_data(
            self.split,
            self.config.zarr_path,
            self.config.spectra_per_plastic * 2,
            patch_size=64,
        )
        self.wn = wn
        self.label_encoding = label_encoding

        s_train, s_val, l_train, l_val = train_test_split(
            spectra, label, test_size=0.5, stratify=label, random_state=42
        )

        self.steps_per_epoch = len(l_train) // self.config.batch_size

        self.train_ds = SpectralDataset(s_train, l_train, self.wn, self.config)
        self.val_ds = SpectralDataset(s_val, l_val, self.wn, self.config)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=None, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=None, pin_memory=True)

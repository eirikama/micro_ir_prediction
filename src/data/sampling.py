import random

import numpy as np
import zarr


def create_experiment_split(zarr_path: str, split_ratio: float = 0.5) -> dict[str, list]:

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
    split: list | None = None,
    zarr_path: str ="/mnt/ssd3/eirik/ProcessedData/microplastics_library.zarr",
    spectra_per_class: int = 8,
    bkg_per_class: int = 4,
    patch_size: int = 128,
    background_max: float = 0.1,
    sample_min: float = 0.5,
    max_class_attempts: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:

    root = zarr.open(zarr_path, mode="r")
    images_group = root["images"]
    wn = np.asarray(root.attrs["wavenumbers"], dtype=np.float32)
    if split is None:
        split = [
            {"name": name, "label": images_group[name].attrs.get("label", name)}
            for name in images_group.keys()
        ]

    unique_labels = sorted(list(set(item["label"] for item in split)))
    rng = np.random.default_rng()

    spectra_per_patch = max(1, spectra_per_class // len(unique_labels))

    all_spectra, all_labels = [], []
    label_encoding = {"bkg": 0}
    for i, label in enumerate(unique_labels):
        names = [item["name"] for item in split if item["label"] == label]
        attempts = 0

        p_sample, p_bkg = [], []
        while len(p_sample) < spectra_per_class or len(p_bkg) < bkg_per_class:
            attempts += 1
            if attempts > max_class_attempts:
                raise RuntimeError(
                    f"\n[Data Sampling Error] Class '{label}' failed to meet quotas after {max_class_attempts} attempts.\n"
                    f"Found: {len(p_sample)}/{spectra_per_class} plastic, {len(p_bkg)}/{bkg_per_class} background.\n"
                    f"Check if background_max ({background_max}) or sample_min ({sample_min}) are too restrictive."
                )

            name = names[rng.integers(len(names))]
            z_arr = images_group[name]["data"]

            H, W, L = z_arr.shape
            y0 = rng.integers(0, max(1, H - patch_size))
            x0 = rng.integers(0, max(1, W - patch_size))

            patch = z_arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
            spectra = patch.reshape(-1, L)
            means = spectra.mean(axis=1)

            spec_mask = means > sample_min
            if spec_mask.any() and len(p_sample) < spectra_per_class:
                valid = spectra[spec_mask]
                k = min(spectra_per_patch, len(valid), spectra_per_class - len(p_sample))
                p_sample.extend(rng.choice(valid, size=k, replace=False))

            bkg_mask = means < background_max
            if bkg_mask.any() and len(p_bkg) < bkg_per_class:
                valid = spectra[bkg_mask]
                k = min(spectra_per_patch, len(valid), bkg_per_class - len(p_bkg))
                p_bkg.extend(rng.choice(valid, size=k, replace=False))

        all_spectra.append(np.stack(p_sample))
        all_spectra.append(np.stack(p_bkg))
        all_labels.append(np.ones(len(p_sample)) * (i + 1))
        all_labels.append(np.zeros(len(p_bkg)))
        label_encoding[label] = i + 1

    return np.hstack(all_labels), np.vstack(all_spectra), wn, label_encoding

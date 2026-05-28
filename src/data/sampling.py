import random
import numpy as np
import zarr
from sklearn.model_selection import train_test_split

def create_experiment_split(zarr_path: str, split_ratio: float = 0.5, seed: int = 42) -> dict[str, list]:
    store = zarr.open(zarr_path, mode="r")
    images_group = store["images"]

    data_list = []
    for name in images_group.keys():
        label = images_group[name].attrs.get("label", "unknown")
        data_list.append({"name": name, "label": label})

    labels = [d["label"] for d in data_list]

    train, test = train_test_split(
        data_list,
        test_size = 1 - split_ratio,
        stratify = labels,
        random_state = seed
    )

    return {"train": train, "test": test}


def get_training_data(
    split: list | None = None,
    zarr_path: str = "/mnt/ssd3/eirik/ProcessedData/microplastics_library.zarr",
    spectra_per_class: int = 8,
    bkg_per_class: int = 4,
    patch_size: int = 128,
    background_max: float = 0.1,
    sample_min: float = 0.5,
    max_class_attempts: int = 1000,
    include_bkg_pixels: bool = True,
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
    label_encoding = {"bkg": 0} if include_bkg_pixels else {}
    for i, label in enumerate(unique_labels):
        class_idx = i + 1 if include_bkg_pixels else i
        label_encoding[label] = class_idx
        names = [item["name"] for item in split if item["label"] == label]
        attempts = 0
        p_sample, p_bkg = [], []
        while len(p_sample) < spectra_per_class or (include_bkg_pixels and len(p_bkg) < bkg_per_class):
            attempts += 1
            if attempts > max_class_attempts:
                bkg_status = f", {len(p_bkg)}/{bkg_per_class} background" if include_bkg_pixels else ""
                raise RuntimeError(
                    f"\n[Data Sampling Error] Class '{label}' failed to meet quotas after {max_class_attempts} attempts.\n"
                    f"Found: {len(p_sample)}/{spectra_per_class} plastic{bkg_status}.\n"
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
            if len(p_sample) < spectra_per_class:
                spec_mask = means > sample_min
                if spec_mask.any():
                    valid = spectra[spec_mask]
                    k = min(spectra_per_patch, len(valid), spectra_per_class - len(p_sample))
                    p_sample.extend(rng.choice(valid, size=k, replace=False))
            if include_bkg_pixels and len(p_bkg) < bkg_per_class:
                bkg_mask = means < background_max
                if bkg_mask.any():
                    valid = spectra[bkg_mask]
                    k = min(spectra_per_patch, len(valid), bkg_per_class - len(p_bkg))
                    p_bkg.extend(rng.choice(valid, size=k, replace=False))
        all_spectra.append(np.stack(p_sample))
        all_labels.append(np.full(len(p_sample), class_idx))
        if include_bkg_pixels:
            all_spectra.append(np.stack(p_bkg))
            all_labels.append(np.zeros(len(p_bkg)))

    return np.hstack(all_labels), np.vstack(all_spectra), wn, label_encoding


def get_test_split(
    test_zarr_path: str,
    train_zarr_path: str,
    rocks: list[str] | None = None,
    conditions: list[str] | None = None,
) -> list[dict]:
    """
    Returns a list of {"name": str, "label": str} dicts for the inference loop.
    One entry per (rock × condition × class) group in the test store.
    """
    root_train = zarr.open(train_zarr_path, mode="r")
    root_test  = zarr.open(test_zarr_path,  mode="r")

    train_classes = list(root_train.attrs["classes"])
    test_classes  = list(root_test.attrs["classes"])
    assert train_classes == test_classes, (
        f"Class mismatch!\n  train: {train_classes}\n  test: {test_classes}"
    )

    test_images = []
    for key in sorted(root_test.group_keys()):
        attrs = root_test[key].attrs
        if rocks      is not None and attrs.get("rock")      not in rocks:      continue
        if conditions is not None and attrs.get("condition") not in conditions: continue

        # Each mineral present in this rock×condition group gets its own entry
        class_counts = attrs.get("class_counts", {})
        for class_name in class_counts:
            test_images.append({
                "name":      key,           # e.g. "gabbro_dusty"
                "label":     class_name,    # e.g. "olivine"
                "rock":      attrs.get("rock"),
                "condition": attrs.get("condition"),
            })

    return test_images

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


def get_training_data_hyperspectral(
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


def create_patient_split(
    zarr_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    classes: list[str] | None = None,  # None = all classes
) -> dict[str, list]:
    """
    Split cores into train / test with no patient appearing in both partitions.

    Args:
        zarr_path:  path to the zarr store
        test_ratio: fraction of patients assigned to test (default 0.2)
        seed:       random seed
        classes:    if provided, only cores containing at least one pixel of
                    these classes are included. Cores with none of the
                    requested classes are dropped entirely.

    Returns dict with keys 'train' and 'test', each a list of dicts:
        {"name": core_name, "label": dominant_class, "patient_id": ...}
    Compatible with get_training_data_pcuk(split=split["train"], classes=classes)
    """
    from collections import defaultdict
    print(zarr_path)
    store        = zarr.open(zarr_path, mode="r")
    images_group = store["images"]
    ALL_CLASSES  = list(store.attrs.get("classes", []))

    # validate requested classes
    if classes is not None:
        unknown = [c for c in classes if c not in ALL_CLASSES]
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}\nValid: {ALL_CLASSES}")
        active_idx = {ALL_CLASSES.index(c) for c in classes}

    # collect cores, optionally filtering to those containing requested classes
    patient_to_cores: dict[str, list] = defaultdict(list)
    skipped = 0

    for name in images_group.keys():
        attrs = images_group[name].attrs
        pid   = attrs.get("patient_id", "unknown")

        if classes is not None:
            y            = images_group[name]["y"][:]
            present_idx  = set(np.unique(y).tolist())
            if not present_idx.intersection(active_idx):
                skipped += 1
                continue
            # dominant class among requested classes only
            mask          = np.isin(y, list(active_idx))
            y_filtered    = y[mask]
            counts        = np.bincount(y_filtered.astype(np.int64), minlength=len(ALL_CLASSES))
            dominant      = ALL_CLASSES[int(counts.argmax())]
        else:
            dominant = attrs.get("label", "unknown")

        patient_to_cores[pid].append({
            "name":       name,
            "label":      dominant,
            "patient_id": pid,
        })

    if skipped:
        print(f"  Skipped {skipped} cores with none of the requested classes")

    patients = list(patient_to_cores.keys())
    rng      = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_test   = max(1, round(len(patients) * test_ratio))
    n_train  = len(patients) - n_test

    train_pats = patients[:n_train]
    test_pats  = patients[n_train:]

    def collect(plist):
        return [core for pid in plist for core in patient_to_cores[pid]]

    split = {"train": collect(train_pats), "test": collect(test_pats)}

    # sanity check
    train_pids = set(c["patient_id"] for c in split["train"])
    test_pids  = set(c["patient_id"] for c in split["test"])
    assert train_pids.isdisjoint(test_pids), "Patient leak: train ∩ test"

    print(
        f"Split -- "
        f"train: {len(train_pats)} patients / {len(split['train'])} cores  |  "
        f"test: {len(test_pats)} patients / {len(split['test'])} cores"
    )
    if classes is not None:
        print(f"  Classes: {classes}")
    print("No patient overlap")

    return split



def get_training_data_spectra(
    split: list | None = None,
    zarr_path: str = "/mnt/ssd3/eirik/ProcessedData/pcuk.zarr",
    spectra_per_class: int | None = None,
    max_per_core_per_class: int | None = None,
    classes: list[str] | None = None,  # None = all 9 classes
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:

    root           = zarr.open(zarr_path, mode="r")
    images_group   = root["images"]
    wn             = np.asarray(root.attrs["wavenumbers"], dtype=np.float32)
    ALL_CLASSES    = list(root.attrs["classes"])

    active_classes = classes if classes is not None else ALL_CLASSES
    unknown        = [c for c in active_classes if c not in ALL_CLASSES]
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}\nValid: {ALL_CLASSES}")

    active_idx     = [ALL_CLASSES.index(c) for c in active_classes]
    label_encoding = {name: i for i, name in enumerate(active_classes)}

    if split is None:
        split = [{"name": name} for name in images_group.keys()]

    rng = np.random.default_rng(seed)
    per_class_X: dict[int, list[np.ndarray]] = {}

    for item in split:
        name = item["name"]
        if name not in images_group:
            print(f"  [warn] {name} not in store — skipping")
            continue
        grp = images_group[name]
        X   = grp["X"][:]
        y   = grp["y"][:]

        for new_idx, orig_idx in enumerate(active_idx):
            mask = y == orig_idx
            if not mask.any():
                continue
            X_cls = X[mask]
            if max_per_core_per_class is not None and len(X_cls) > max_per_core_per_class:
                X_cls = X_cls[rng.choice(len(X_cls), max_per_core_per_class, replace=False)]
            per_class_X.setdefault(new_idx, []).append(X_cls)

    pooled = {
        new_idx: np.concatenate(chunks, axis=0)
        for new_idx, chunks in per_class_X.items()
    }

    missing = [active_classes[i] for i in range(len(active_classes)) if i not in pooled]
    if missing:
        print(f"  [warn] No pixels found for: {missing}")

    cap = min(len(X_cls) for X_cls in pooled.values())
    if spectra_per_class is not None:
        cap = min(cap, spectra_per_class)

    all_X, all_y = [], []
    print(f"\n{'Class':<45} {'Available':>10} {'Sampled':>8}")
    print("-" * 65)
    for new_idx, name in enumerate(active_classes):
        if new_idx not in pooled:
            print(f"  {name:<43} {'MISSING':>10}")
            continue
        X_cls = pooled[new_idx]
        sel   = rng.choice(len(X_cls), cap, replace=False)
        all_X.append(X_cls[sel])
        all_y.append(np.full(cap, new_idx, dtype=np.int64))
        print(f"  {name:<43} {len(X_cls):>10,} {cap:>8,}")

    print(f"\n  Balanced at {cap:,} px/class  ->  {cap * len(all_X):,} total\n")

    X_out = np.vstack(all_X)
    y_out = np.hstack(all_y)
    order = rng.permutation(len(X_out))
    return y_out[order], X_out[order], wn, label_encoding


def get_test_split(test_zarr_path, train_zarr_path, rocks=None, conditions=None):
    root_train    = zarr.open(train_zarr_path, mode="r")
    root_test     = zarr.open(test_zarr_path,  mode="r")
    train_classes = list(root_train.attrs["classes"])
    test_classes  = list(root_test.attrs["classes"])
    assert train_classes == test_classes, (
        f"Class mismatch!\n  train: {train_classes}\n  test:  {test_classes}"
    )
    test_images = []
    for key in sorted(root_test.group_keys()):
        grp   = root_test[key]
        attrs = grp.attrs
        if rocks      is not None and attrs.get("rock")      not in rocks:      continue
        if conditions is not None and attrs.get("condition") not in conditions: continue
        test_images.append({
            "name":      key,
            "label":     attrs.get("rock", key),   # falls back to key for bacteria
            "rock":      attrs.get("rock"),
            "condition": attrs.get("condition"),
        })
    return test_images

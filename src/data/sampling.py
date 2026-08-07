import random
import numpy as np
import zarr
from sklearn.model_selection import train_test_split


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


def get_training_data_hyperspectral(
    split: list | None = None,
    zarr_path: str = "/mnt/ssd3/eirik/ProcessedData/microplastics_library.zarr",
    spectra_per_class: int = 8,
    bkg_per_class: int = 4,
    background_max: float = 0.1,
    sample_min: float = 0.5,
    include_bkg_pixels: bool = True,
    seed: int = 42,
    patch_size: int = 128,        # accepted-and-ignored (kept so configs don't break)
    max_class_attempts: int = 1000,  # accepted-and-ignored (no rejection loop anymore)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    root         = zarr.open(zarr_path, mode="r")
    images_group = root["images"]
    wn           = np.asarray(root.attrs["wavenumbers"], dtype=np.float32)

    if split is None:
        split = [
            {"name": name, "label": images_group[name].attrs.get("label", name)}
            for name in images_group.keys()
        ]

    unique_labels = sorted(set(item["label"] for item in split))
    rng           = np.random.default_rng(seed)

    label_encoding = {"bkg": 0} if include_bkg_pixels else {}
    all_spectra, all_labels = [], []

    header = f"\n{'Class':<45} {'Imgs':>5} {'Avail':>10} {'Sampled':>8}"
    if include_bkg_pixels:
        header += f" {'BkgAvail':>9} {'BkgSmp':>7}"
    print(header)
    print("-" * len(header))

    for i, label in enumerate(unique_labels):
        class_idx = i + 1 if include_bkg_pixels else i
        label_encoding[label] = class_idx
        names = [item["name"] for item in split if item["label"] == label]

        # ── pass 1: read each image once, split pixels by threshold ──────────
        sample_cores, bkg_cores = [], []
        for name in names:
            if name not in images_group:
                print(f"  [warn] {name} not in store — skipping")
                continue
            data    = images_group[name]["data"][:]          # (H, W, L) full cube
            L       = data.shape[-1]
            spectra = data.reshape(-1, L)
            means   = spectra.mean(axis=1)

            s_mask = means > sample_min
            if s_mask.any():
                sample_cores.append(spectra[s_mask])
            if include_bkg_pixels:
                b_mask = means < background_max
                if b_mask.any():
                    bkg_cores.append(spectra[b_mask])

        # ── pass 2: water-fill the budgets across images ─────────────────────
        s_avail  = sum(len(c) for c in sample_cores)
        s_budget = min(spectra_per_class, s_avail)
        s_quotas = allocate([len(c) for c in sample_cores], s_budget, rng)
        s_sel    = [c[rng.choice(len(c), int(q), replace=False)]
                    for c, q in zip(sample_cores, s_quotas) if q > 0]

        if not s_sel:
            raise RuntimeError(
                f"\n[Data Sampling Error] Class '{label}' has no sample pixels "
                f"(0/{spectra_per_class} found).\n"
                f"Check if background_max ({background_max}) or sample_min "
                f"({sample_min}) are too restrictive."
            )

        X_s = np.concatenate(s_sel, axis=0)
        all_spectra.append(X_s)
        all_labels.append(np.full(len(X_s), class_idx, dtype=np.int64))

        line = f"  {label:<43} {len(names):>5} {s_avail:>10,} {len(X_s):>8,}"

        if include_bkg_pixels:
            b_avail  = sum(len(c) for c in bkg_cores)
            b_budget = min(bkg_per_class, b_avail)
            b_quotas = allocate([len(c) for c in bkg_cores], b_budget, rng)
            b_sel    = [c[rng.choice(len(c), int(q), replace=False)]
                        for c, q in zip(bkg_cores, b_quotas) if q > 0]
            if b_sel:
                X_b = np.concatenate(b_sel, axis=0)
                all_spectra.append(X_b)
                all_labels.append(np.zeros(len(X_b), dtype=np.int64))
                line += f" {b_avail:>9,} {len(X_b):>7,}"
            else:
                line += f" {b_avail:>9,} {0:>7}"

        print(line)

    print(f"\n  Total: {sum(len(y) for y in all_labels):,} spectra "
          f"across {len(unique_labels)} classes\n")

    return np.hstack(all_labels), np.vstack(all_spectra), wn, label_encoding



def allocate(capacities, budget, rng):
    """Water-filling allocation.

    Distribute `budget` picks across cores without exceeding any core's
    capacity, redistributing the shortfall from small/full cores to cores
    that still have room. Returns an int array of per-core quotas summing
    to min(budget, total capacity).
    """
    caps   = np.asarray(capacities, dtype=np.int64)
    budget = min(int(budget), int(caps.sum()))
    quota  = np.zeros_like(caps)

    while budget > 0:
        active = quota < caps                       # cores with room left
        n = int(active.sum())
        if n == 0:
            break                                   # everything is full
        give = budget // n
        if give == 0:
            # fewer picks left than active cores: hand out one each,
            # to a random subset, so no core is systematically favoured
            idx = rng.permutation(np.where(active)[0])[:budget]
            quota[idx] += 1
            budget -= len(idx)
            continue
        add = np.where(active, np.minimum(give, caps - quota), 0)
        quota += add
        budget -= int(add.sum())

    return quota


def get_training_data_spectra(
    split: list | None = None,
    zarr_path: str = "/mnt/ssd3/eirik/ProcessedData/pcuk.zarr",
    spectra_per_class: int | None = None,
    classes: list[str] | None = None,
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

    # per_class_cores[new_idx] = list of np.ndarray, one per core
    per_class_cores: dict[int, list[np.ndarray]] = {}

    for item in split:
        name = item["name"]
        if name not in images_group:
            print(f"  [warn] {name} not in store — skipping")
            continue
        grp = images_group[name]
        if "X" in grp and "y" in grp:
            # per-core store: pixels carry mixed labels, filter by y
            X = grp["X"][:]
            y = grp["y"][:]
            for new_idx, orig_idx in enumerate(active_idx):
                mask = y == orig_idx
                if not mask.any():
                    continue
                per_class_cores.setdefault(new_idx, []).append(X[mask])
        elif "data" in grp:
            # per-class library store: the whole group is one class (e.g. bacteria
            # reference library) — no per-pixel y labels to mask against.
            label = grp.attrs.get("label", name)
            if label not in active_classes:
                continue
            new_idx = active_classes.index(label)
            X = grp["data"][:]
            X = X.reshape(X.shape[0], -1)  # collapse any singleton channel dims
            per_class_cores.setdefault(new_idx, []).append(X)
        else:
            print(f"  [warn] {name} has neither X/y nor data — skipping")

    missing = [active_classes[i] for i in range(len(active_classes)) if i not in per_class_cores]
    if missing:
        print(f"  [warn] No pixels found for: {missing}")

    all_X, all_y = [], []
    print(f"\n{'Class':<45} {'Cores':>6} {'Available':>10} {'Sampled':>8}")
    print("-" * 73)

    for new_idx, name in enumerate(active_classes):
        if new_idx not in per_class_cores:
            print(f"  {name:<43} {'MISSING':>6}")
            continue

        cores       = per_class_cores[new_idx]
        n_cores     = len(cores)
        total_avail = sum(len(c) for c in cores)

        # target budget for this class
        budget = total_avail if spectra_per_class is None \
                 else min(spectra_per_class, total_avail)

        # water-filling: never leaves usable pixels on the table
        quotas   = allocate([len(c) for c in cores], budget, rng)

        selected = []
        for ci, q in enumerate(quotas):
            if q > 0:
                idx = rng.choice(len(cores[ci]), int(q), replace=False)
                selected.append(cores[ci][idx])

        X_cls   = np.concatenate(selected, axis=0)
        sampled = len(X_cls)
        all_X.append(X_cls)
        all_y.append(np.full(sampled, new_idx, dtype=np.int64))
        print(f"  {name:<43} {n_cores:>6} {total_avail:>10,} {sampled:>8,}")

    print(f"\n  Total: {sum(len(y) for y in all_y):,} spectra across {len(all_X)} classes\n")

    X_out = np.vstack(all_X)
    y_out = np.hstack(all_y)
    order = rng.permutation(len(X_out))
    return y_out[order], X_out[order], wn, label_encoding

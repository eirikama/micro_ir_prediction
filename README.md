# Spectral Classification Pipeline

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Zarr](https://img.shields.io/badge/Zarr-LMDB-orange)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)

Deep learning pipeline for classification of both spectral data and hyperspectral microscopy images across multiple material domains. Trains a 1D attention-augmented convolutional neural network (AACNN) on synthetically augmented reference spectra and runs GPU-accelerated pixel-wise inference over full hyperspectral image cubes, producing per-pixel class probability maps.

**Supported domains:**

| Domain | Modality | Augmentations |
|---|---|---|
| `microplastic` | IR (FTIR) | Mie scattering, polynomial baseline, noise, CO₂ peaks |
| `pollen` | IR (FTIR) | Mie scattering, polynomial baseline, noise, CO₂ peaks |
| `textile` | IR (FTIR) | Mie scattering, polynomial baseline, noise |
| `mlrod` | Raman | Cosmic ray spikes, fluorescence background (PCA-fitted), shot noise |
| `bacteria` | Raman | Cosmic ray spikes, fluorescence background (PCA-fitted), shot noise |

---

## Overview

Each domain has its own config subtree under `configs/domain/`. All domains share the same model architecture and training loop; only the data paths, augmentation pipeline, and class count differ.

The pipeline:

1. Samples reference spectra from Zarr image cubes with balanced per-class sampling
2. Applies domain-appropriate synthetic augmentation on every training batch (online, not pre-computed)
3. Trains an AACNN with focal loss, mixed-precision, and early stopping
4. Runs pixel-wise inference over held-out image cubes, saving per-pixel probability maps
5. Tracks all experiments (params, metrics, checkpoints, git state) with MLflow

Validation can be done two ways, controlled by `data.intrinsic_validation`:
- `True` — stratified train/test split from the single training zarr
- `False` — train on the full training zarr, evaluate on a separate test zarr

---

## Project structure

```
micro_ir_prediction/
├── Dockerfile
├── docker-compose.yml
├── docker/
│   └── sqlite-autoconf-3450200.tar.gz   # bundled for offline Docker build
├── .env                                 # PROJECT_DIR path (not committed)
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
├── main.py                              # entry point
│
├── configs/
│   ├── config.yaml                      # root config (domain selector, mlflow, mode, seed)
│   └── domain/
│       ├── microplastic.yaml            # domain-level overrides (mlflow experiment name, etc.)
│       ├── microplastic/
│       │   ├── data/default.yaml        # zarr paths, sampling, IR augmentations
│       │   ├── model/aacnn.yaml         # num_classes, lr, architecture
│       │   ├── trainer/default.yaml     # epochs, patience, precision
│       │   └── inference/default.yaml  # batch size, pred_store_path, ckpt_path
│       ├── pollen.yaml                  # (same layout as microplastic/)
│       ├── pollen/
│       ├── textile.yaml
│       ├── textile/
│       ├── mlrod.yaml                   # Raman domain
│       └── mlrod/
│           ├── data/default.yaml        # Raman zarr paths, cosmic ray / fluorescence / shot noise
│           ├── model/aacnn.yaml
│           ├── trainer/default.yaml
│           └── inference/default.yaml
│
└── src/
    ├── configs/
    │   └── config_schema.py             # typed dataclass config definitions
    ├── data/
    │   ├── augmentation.py              # IR and Raman augmentation registry
    │   ├── datamodule.py                # SpectralDataModule + balanced IterableDataset
    │   └── sampling.py                  # zarr patch sampling, train/test split helpers
    ├── models/
    │   ├── aacnn.py                     # AACNN LightningModule
    │   ├── blocks.py                    # attention-augmented conv blocks
    │   └── loss.py                      # FocalLoss
    ├── training/
    │   ├── callbacks.py                 # ExtendedLogger (overfit gap, grad norm, LR)
    │   └── trainer_engine.py            # Trainer setup and fit
    ├── inference/
    │   ├── inference_engine.py          # GPU-accelerated pixel-wise inference
    │   └── export_inference.py          # Zarr/LMDB output saving
    ├── physics/
    │   ├── sphere/
    │   │   ├── sphere_mie.pyx           # Cython spherical Mie scattering
    │   │   └── setup_sphere_mie.py
    │   └── cylinder/
    │       ├── bessel.pyx               # Cython Bessel functions
    │       ├── cylinder_mie.py          # cylindrical Mie scattering
    │       └── setup_bessel.py
    └── utils.py
```

---

## Data format

### Training zarr

```
{domain}_library.zarr/
    attrs:
        wavenumbers    list[float]   spectral axis
        classes        list[str]     class names
    images/
        {image_name}/
            data       (H, W, Bands)   float32
            attrs:     label, class_counts, ...
```

### Test zarr (when `intrinsic_validation: False`)

```
{domain}_test.zarr/
    attrs:
        wavenumbers    list[float]
        classes        list[str]     must match training zarr
    {image_name}/
        data           (H, W, Bands)   float32
        y              (H*W,)          int     per-pixel class labels
        attrs:         rock, condition, class_counts, ...
```

---

## Installation

### Docker (recommended)

```bash
# bundled SQLite for offline build (only needed once)
mkdir -p docker
wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz -P docker/

# set your project directory
echo 'PROJECT_DIR=/path/to/micro_ir_prediction' > .env

# build image (compiles Cython extensions inside the container)
docker compose build
```

### Local development (without Docker)

```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt

# compile Cython Mie scattering extensions
cd src/physics/sphere  && python setup_sphere_mie.py build_ext --inplace && cd ../../..
cd src/physics/cylinder && python setup_bessel.py    build_ext --inplace && cd ../../..
```

---

## Configuration

All parameters are controlled via Hydra. Select a domain with `domain=<name>`; every other setting can be overridden on the command line with `key=value`.

### Key parameters

| Parameter | Default location | Description |
|---|---|---|
| `domain` | `config.yaml` | `microplastic`, `pollen`, `textile`, `mlrod` |
| `mode` | `config.yaml` | `train`, `infer`, or `all` |
| `seed` | `config.yaml` | global random seed |
| `data.spectra_per_class` | `domain/.../data/default.yaml` | reference spectra sampled per class |
| `data.batch_size` | `domain/.../data/default.yaml` | training batch size |
| `data.intrinsic_validation` | `domain/.../data/default.yaml` | `True` = split training zarr; `False` = use separate test zarr |
| `data.augment_train` | `domain/.../data/default.yaml` | enable augmentation during training |
| `data.z_normalize` | `domain/.../data/default.yaml` | per-spectrum z-score normalisation |
| `trainer.max_epochs` | `domain/.../trainer/default.yaml` | maximum training epochs |
| `trainer.early_stopping_patience` | `domain/.../trainer/default.yaml` | early stopping patience (epochs) |
| `inference.ckpt_path` | `domain/.../inference/default.yaml` | checkpoint to load for inference |
| `inference.pred_store_path` | `domain/.../inference/default.yaml` | output LMDB path |
| `mlflow.tracking_uri` | `config.yaml` | MLflow database URI |

---

## Running

All services are launched via Docker Compose. Set `PROJECT_DIR` in `.env` first.

### Train + infer (full run)

```bash
docker compose run --rm train python main.py \
    domain=mlrod \
    mode=all \
    seed=0
```

### Override data parameters

```bash
docker compose run --rm train python main.py \
    domain=microplastic \
    mode=all \
    seed=1 \
    data.spectra_per_class=128 \
    data.batch_size=64
```

### Inference only (from existing checkpoint)

```bash
docker compose run --rm train python main.py \
    domain=microplastic \
    mode=infer \
    inference.ckpt_path=/app/checkpoints/best-epoch=42-val_acc=0.9123.ckpt
```

### Hyperparameter sweep (Hydra multirun)

```bash
docker compose run --rm train python main.py --multirun \
    domain=mlrod \
    data.spectra_per_class=8,16,32,64,128 \
    seed="range(0,5)" \
    mode=all
```

### MLflow UI

```bash
docker compose up mlflow-ui -d
# http://localhost:5000
```

### Jupyter

```bash
docker compose up jupyter -d
# http://localhost:8888
```

### SSH port forwarding (PuTTY)

```
Connection → SSH → Tunnels
  Source port: 5000   Destination: localhost:5000   → MLflow UI
  Source port: 8888   Destination: localhost:8888   → Jupyter
```

---

## Augmentations

Augmentations are applied online inside each training batch. The pipeline is configured per domain under `data.augmentations` and dispatched through `AUG_REGISTRY`.

### IR domains (microplastic, pollen, textile)

| Type | Description |
|---|---|
| `mie_scattering` | Spherical or cylindrical Mie scattering (Cython, physically parameterised) |
| `polynomial_baseline` | Additive polynomial baseline (signal or background pixels separately) |
| `noise` | Additive white Gaussian noise |
| `co2_peaks` | Synthetic CO₂ absorption peaks |

### Raman domain (mlrod, bacteria)

| Type | Description |
|---|---|
| `cosmic_rays` | Random narrow high-amplitude spikes (Poisson-rate, triangle profile) |
| `fluorescence` | PCA-fitted baseline — fit once on training spectra before training starts |
| `shot_noise` | Signal-proportional Gaussian noise |

The `fluorescence` augmentor is fitted in `main.py` before training and cached; the cache is keyed by spectral length so multi-domain runs are safe.

---

## Inference outputs

Predictions are saved to an LMDB-backed Zarr store:

```
predictions.lmdb/
    N{spectra_per_class}_seed{seed}/
        {image_name}/
            argmax_map      (H, W)      uint8
            bg_prob         (H, W)      float16
            top_k_classes   (H, W, k)   uint8
            top_k_probs     (H, W, k)   float16
            attrs:  N, seed, true_idx, hparams, ...
```

Load and analyse:

```python
import zarr, numpy as np

store  = zarr.open(zarr.LMDBStore("predictions.lmdb", readonly=True), mode="r")
grp    = store["N64_seed0/image_name"]
argmax = grp["argmax_map"][:]
bg_prob = grp["bg_prob"][:]

mask = bg_prob.flatten() <= 0.5
acc  = np.mean(argmax.flatten()[mask] == grp.attrs["true_idx"])
```

Sweep thresholds across all trials:

```python
import pandas as pd

results = []
for trial_id in store.keys():
    for image_name in store[trial_id].keys():
        grp      = store[f"{trial_id}/{image_name}"]
        bg_prob  = grp["bg_prob"][:].flatten()
        argmax   = grp["argmax_map"][:].flatten()
        true_idx = grp.attrs["true_idx"]
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            mask = bg_prob <= threshold
            acc  = float(np.mean(argmax[mask] == true_idx)) if mask.sum() > 0 else float("nan")
            results.append({"N": grp.attrs["N"], "seed": grp.attrs["seed"],
                            "image": image_name, "threshold": threshold, "accuracy": acc})

df = pd.DataFrame(results)
print(df.groupby(["N", "threshold"])["accuracy"].agg(["mean", "std"]).round(4))
```

---

## Cython extensions

Two physics modules must be compiled before use (done automatically inside Docker):

```bash
# spherical Mie scattering
cd src/physics/sphere
python setup_sphere_mie.py build_ext --inplace

# cylindrical Mie scattering (Bessel functions)
cd src/physics/cylinder
python setup_bessel.py build_ext --inplace
```

These are only required for IR domains. The Raman (mlrod) domain does not call them.

---

## License

MIT License — see `LICENSE` for details.

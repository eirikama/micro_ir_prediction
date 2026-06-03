# Spectral Classification Pipeline

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa?logo=python&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.x-2979FF?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Zarr](https://img.shields.io/badge/Zarr-LMDB-orange)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)

Deep learning pipeline for classification of spectral data and hyperspectral microscopy images across multiple material domains. Trains a 1D attention-augmented convolutional neural network (AACNN) on synthetically augmented reference spectra and runs GPU-accelerated pixel-wise inference over full hyperspectral image cubes, producing per-pixel class probability maps.

**Supported domains:**

| Domain | Modality | Active augmentations |
|---|---|---|
| `microplastic` | IR (FTIR) | Mie scattering, polynomial baseline (signal + background), noise |
| `textile` | IR (FTIR) | Mie scattering, polynomial baseline (signal + background), noise |
| `pollen` | IR (FTIR) | None by default (all disabled; explore via sweep) |
| `mlrod` | Raman | Cosmic ray spikes, fluorescence background (PCA-fitted), shot noise |
| `bacteria` | Raman | Fluorescence background (PCA-fitted), shot noise |

---

## Overview

Each domain has its own config subtree under `configs/domain/`. All domains share the same model architecture and training loop; only the data paths, augmentation pipeline, and class count differ.

The pipeline:

1. Samples reference spectra from Zarr image cubes with balanced per-class sampling
2. Applies domain-appropriate synthetic augmentation on every training batch (online, fresh each epoch)
3. Trains an AACNN with focal loss, mixed-precision, and early stopping
4. Runs pixel-wise inference over held-out image cubes, saving per-pixel probability maps
5. Tracks all experiments (params, metrics, checkpoints, git state) with MLflow

Validation is controlled by `data.intrinsic_validation`:
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
├── requirements.txt
├── main.py                              # entry point — returns val_acc for Optuna
├── sweep.py                             # convenience CLI for hyperparameter sweeps
│
├── configs/
│   ├── config.yaml                      # root config (domain, mlflow, mode, seed)
│   ├── hydra/
│   │   ├── sweeper/                     # Optuna sweeper configs (one per domain)
│   │   │   ├── optuna_microplastic.yaml
│   │   │   ├── optuna_textile.yaml
│   │   │   ├── optuna_pollen.yaml
│   │   │   ├── optuna_mlrod.yaml
│   │   │   └── optuna_bacteria.yaml
│   │   └── launcher/
│   │       └── submitit_slurm.yaml      # SLURM launcher (cluster use)
│   └── domain/
│       ├── microplastic.yaml            # MLflow experiment name, etc.
│       └── microplastic/
│           ├── data/default.yaml        # zarr paths, sampling, augmentations (dict)
│           ├── model/aacnn.yaml         # num_classes, lr, architecture
│           ├── trainer/default.yaml     # epochs, patience, precision
│           └── inference/default.yaml  # batch size, pred_store_path, ckpt_path
│           (same layout for pollen/, textile/, mlrod/, bacteria/)
│
├── slurm/
│   ├── train.sh                         # single training job (Singularity)
│   └── sweep.sh                         # parallel sweep job array (Singularity)
│
└── src/
    ├── configs/config_schema.py         # typed dataclass config definitions
    ├── data/
    │   ├── augmentation.py              # IR and Raman augmentation registry
    │   ├── datamodule.py                # SpectralDataModule; fluorescence fit in setup()
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
    └── physics/
        ├── sphere/sphere_mie.pyx        # Cython spherical Mie scattering
        └── cylinder/cylinder_mie.py     # cylindrical Mie scattering
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

### Docker (recommended for local use)

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

Requires Python 3.8+. Python 3.10+ recommended (available via `deadsnakes` PPA on Ubuntu or `conda`).

```bash
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt

# compile Cython Mie scattering extensions (IR domains only)
cd src/physics/sphere  && python setup_sphere_mie.py build_ext --inplace && cd ../../..
cd src/physics/cylinder && python setup_bessel.py    build_ext --inplace && cd ../../..
```

---

## Configuration

All parameters are controlled via Hydra. Select a domain with `domain=<name>`; any parameter can be overridden with `key=value` on the command line.

### Key parameters

| Parameter | Default location | Description |
|---|---|---|
| `domain` | `config.yaml` | `microplastic`, `pollen`, `textile`, `mlrod`, `bacteria` |
| `mode` | `config.yaml` | `train`, `infer`, or `all` |
| `seed` | `config.yaml` | global random seed |
| `data.spectra_per_class` | `domain/.../data/default.yaml` | reference spectra sampled per class |
| `data.batch_size` | `domain/.../data/default.yaml` | training batch size |
| `data.intrinsic_validation` | `domain/.../data/default.yaml` | `True` = split training zarr; `False` = separate test zarr |
| `data.augment_train` | `domain/.../data/default.yaml` | enable online augmentation during training |
| `data.z_normalize` | `domain/.../data/default.yaml` | per-spectrum z-score normalisation |
| `trainer.max_epochs` | `domain/.../trainer/default.yaml` | maximum training epochs |
| `trainer.early_stopping_patience` | `domain/.../trainer/default.yaml` | early stopping patience (epochs) |
| `inference.ckpt_path` | `domain/.../inference/default.yaml` | checkpoint path for inference |
| `mlflow.tracking_uri` | `config.yaml` | MLflow database URI |

### Augmentation config format

Augmentations are configured as a **named dict** under `data.augmentations`, not a list. The key is the augmentation name used for Hydra overrides; if it differs from the registry key (e.g. `polynomial_baseline_signal`), a `type:` field specifies the registry dispatch name.

```yaml
# configs/domain/bacteria/data/default.yaml
augmentations:
  fluorescence:             # key == registry key, no `type:` needed
    enabled: true
    ratio: 0.5
    amplitude_min: 0.0
    amplitude_max: 0.05

  shot_noise:
    enabled: true
    ratio: 0.5
    scale: 0.005

# configs/domain/microplastic/data/default.yaml
augmentations:
  mie_scattering:
    enabled: true
    ratio: 0.5
    ...

  polynomial_baseline_signal:   # two entries of the same type
    type: polynomial_baseline   # explicit dispatch key
    signal_only: true
    ratio: 0.6

  polynomial_baseline_background:
    type: polynomial_baseline
    background_only: true
    ratio: 0.6
```

Individual parameters can be overridden directly on the command line:

```bash
python main.py domain=bacteria \
    data.augmentations.fluorescence.amplitude_max=0.1 \
    data.augmentations.shot_noise.scale=0.01
```

---

## Running

### Train + infer (full run)

```bash
docker compose run --rm train python main.py domain=mlrod mode=all seed=0
```

### Override data parameters

```bash
docker compose run --rm train python main.py \
    domain=microplastic mode=all seed=1 \
    data.spectra_per_class=128 data.batch_size=64
```

### Inference only (from existing checkpoint)

```bash
docker compose run --rm train python main.py \
    domain=microplastic mode=infer \
    inference.ckpt_path=/app/checkpoints/best-epoch=42-val_acc=0.9123.ckpt
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

### SSH port forwarding (PuTTY / ssh -L)

```
Source port 5000 → localhost:5000   (MLflow UI)
Source port 8888 → localhost:8888   (Jupyter)
```

---

## Hyperparameter sweeps

Sweeps use the **Hydra Optuna Sweeper** plugin. The search space for each domain lives in `configs/hydra/sweeper/optuna_<domain>.yaml`. `main.py` returns `val_acc` as the Optuna objective.

Three groups of parameters are swept simultaneously:
- **Training** — learning rate, weight decay, batch size
- **Architecture** — conv channels, kernel size, dropout, focal loss gamma
- **Augmentation** — per-augmentation `ratio`, intensity, and physics parameters

Augmentation *types* are never swept — only their numerical parameters.

### Install sweep dependencies (once)

```bash
pip install hydra-optuna-sweeper hydra-submitit-launcher
# already in requirements.txt — included automatically in Docker
```

### Run a sweep locally

```bash
# convenience wrapper (100 trials from sweep config):
python sweep.py --domain bacteria

# with SQLite persistence (can resume after interruption):
python sweep.py --domain bacteria --storage sqlite:///bacteria_sweep.db

# override number of trials:
python sweep.py --domain microplastic --n-trials 50

# equivalent raw Hydra command:
python main.py --multirun domain=bacteria hydra/sweeper=optuna_bacteria mode=train
```

### Inspect results

```python
import optuna

study = optuna.load_study(
    study_name="bacteria_augmentation_sweep",
    storage="sqlite:///bacteria_sweep.db",
)
print(f"Best val_acc: {study.best_value:.4f}")
print(study.best_params)
```

### Adding or removing sweep parameters

Edit the relevant `configs/hydra/sweeper/optuna_<domain>.yaml`. Add a line to include a parameter, delete or comment it out to exclude it. The format follows Optuna's override syntax:

```yaml
params:
  model.lr: tag(log, interval(5e-6, 5e-4))      # log-uniform float
  data.batch_size: choice(32, 64, 128)           # categorical
  model.pred_dropout: interval(0.0, 0.5)         # uniform float
  data.augmentations.fluorescence.amplitude_max: tag(log, interval(0.005, 0.3))
```

---

## Running on a cluster (SLURM + Singularity)

HPC clusters do not support Docker (requires root). Use **Singularity/Apptainer** instead — it converts your Docker image with zero code changes.

### Build the Singularity image (one-time)

```bash
# Option A: pull from a container registry
singularity pull micro_ir.sif docker://ghcr.io/youruser/micro_ir:latest

# Option B: convert from a local Docker image (no registry needed)
docker save micro_ir:latest | gzip > micro_ir.tar.gz
scp micro_ir.tar.gz cluster:/scratch/$USER/
# on the cluster:
singularity build micro_ir.sif docker-archive://micro_ir.tar.gz
```

### Single training job

```bash
# edit SIF, DATA_DIR, and #SBATCH --partition in the script first
sbatch slurm/train.sh

# override domain:
sbatch --export=ALL,DOMAIN=bacteria slurm/train.sh
```

### Parallel sweep (job array)

Each array element runs exactly one Optuna trial. All elements share a SQLite file and Optuna's TPE sampler coordinates without conflicts.

```bash
# 50 trials, max 4 running simultaneously:
sbatch --array=0-49%4 --export=ALL,DOMAIN=bacteria slurm/sweep.sh

# 100 trials, all parallel:
sbatch --array=0-99 slurm/sweep.sh
```

Before submitting, edit `slurm/sweep.sh` to set:
- `SIF=` — path to your `.sif` file
- `DATA_DIR=` — where your zarr files live
- `#SBATCH --partition=` — your cluster's GPU partition name

---

## Augmentations

Augmentations are applied online inside each training batch, dispatched through `AUG_REGISTRY` in `src/data/augmentation.py`. The `fluorescence` PCA basis is fitted once inside `SpectralDataModule.setup()` before training starts.

### IR domains (microplastic, textile, pollen)

| Registry key | Config key(s) | Description |
|---|---|---|
| `mie_scattering` | `mie_scattering` | Spherical or cylindrical Mie scattering (Cython, physically parameterised) |
| `polynomial_baseline` | `polynomial_baseline_signal`, `polynomial_baseline_background` | Additive polynomial baseline, applied to signal and background pixels separately |
| `noise` | `noise` | Additive white Gaussian noise |
| `co2_peaks` | `co2_peaks` | Synthetic CO₂ absorption peaks |

### Raman domains (mlrod, bacteria)

| Registry key | Config key | Description |
|---|---|---|
| `cosmic_rays` | `cosmic_rays` | Random narrow high-amplitude spikes (Poisson rate, triangle profile) |
| `fluorescence` | `fluorescence` | PCA-fitted additive baseline; fitted once per `setup()` call |
| `shot_noise` | `shot_noise` | Signal-proportional Gaussian noise |

Amplitude ranges for `fluorescence` and `cosmic_rays` are configured as `amplitude_min` / `amplitude_max` (separate fields, directly addressable by Hydra overrides).

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

store   = zarr.open(zarr.LMDBStore("predictions.lmdb", readonly=True), mode="r")
grp     = store["N64_seed0/image_name"]
argmax  = grp["argmax_map"][:]
bg_prob = grp["bg_prob"][:]

mask = bg_prob.flatten() <= 0.5
acc  = np.mean(argmax.flatten()[mask] == grp.attrs["true_idx"])
```

---

## Cython extensions

Two physics modules must be compiled before use (done automatically inside Docker):

```bash
cd src/physics/sphere  && python setup_sphere_mie.py build_ext --inplace
cd src/physics/cylinder && python setup_bessel.py    build_ext --inplace
```

Required for IR domains only. Raman domains (mlrod, bacteria) do not call them.

---

## License

MIT License — see `LICENSE` for details.

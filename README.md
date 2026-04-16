# Inrefred Hyperspectral Prediction

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Zarr](https://img.shields.io/badge/Zarr-LMDB-orange)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)

Deep learning pipeline for automated classification of microplastic particle types from hyperspectral FTIR microscopy images. Trains a 1D attention-augmented convolutional neural network (AACNN) on synthetically augmented reference spectra and runs GPU-accelerated pixel-wise inference over full hyperspectral image cubes, producing per-pixel class probability maps for downstream analysis.

---

## Overview

Hyperspectral FTIR microscopy produces images where each pixel contains a full infrared absorption spectrum. This pipeline:

1. Trains a spectral classifier on synthetically augmented reference spectra with configurable Mie scattering, polynomial background, and noise augmentation
2. Runs multi-GPU pixel-wise inference over full `(H, W, Bands)` hyperspectral cubes stored in Zarr format
3. Saves per-pixel probability maps, argmax predictions, and top-k class scores to a compressed LMDB-backed Zarr store
4. Tracks all experiments with MLflow and supports full hyperparameter sweeps via Hydra multirun
5. Runs reproducibly inside Docker with GPU passthrough

---

## Project structure

```
microscopy_prediction/
├── Dockerfile
├── docker-compose.yml
├── .env                           # PROJECT_DIR path (not committed)
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
├── main.py                           # entry point
│
├── configs/
│   ├── config.yaml                   # root config
│   ├── data/
│   │   └── default.yaml              # data and augmentation config
│   ├── model/
│   │   └── aacnn.yaml                # model architecture config
│   └── trainer/
│       └── default.yaml              # PyTorch Lightning trainer config
│
├── src/
│   ├── configs/
│   │   └── config_schema.py          # typed dataclass config definitions
│   ├── data/
│   │   ├── augmentation.py           # spectral augmentation
│   │   ├── datamodule.py             # SpectralDataModule, experiment split
│   │   └── sampling.py               # data sampling utilities
│   ├── models/
│   │   ├── aacnn.py                  # AACNN LightningModule
│   │   ├── blocks.py                 # attention-augmented conv blocks
│   │   └── loss.py                   # FocalLoss
│   ├── training/
│   │   ├── callbacks.py              # custom Lightning callbacks
│   │   └── trainer_engine.py         # training loop
│   ├── inference/
│   │   ├── inference_engine.py       # multi-GPU inference workers
│   │   └── export_inference.py       # Zarr/LMDB output saving
│   ├── physics/
│   │   ├── mie.pyx                   # Cython Mie scattering implementation
│   │   └── setup_mie.py              # Cython build script
│   └── utils.py                      # shared utilities
│
├── notebooks/
│   └── visualize_results.ipynb
└── docker/
    └── sqlite-autoconf-3450200.tar.gz   # bundled for offline Docker build
```

---

## Data format

Input data is a Zarr archive with the following layout:

```
microplastics_library.zarr/
    images/
        {image_name}/
            data    (H, W, Bands)   float32
            attrs:  label, dimensions, mean_spectrum
```

Image dimensions are typically powers of 2 (`2^k × 2^l`). Bands is the number of spectral channels.

---

## Installation

### Docker (recommended)

```bash
# copy the sqlite tarball for offline build (only needed once)
mkdir -p docker
wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz -P docker/

# set your project directory
echo 'PROJECT_DIR=/path/to/microscopy_prediction' > .env

# build image
docker compose build
```

### Local development (without Docker)

```bash
git clone https://github.com/yourname/microplastics-predict.git
cd microplastics-predict

python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt

# build Cython Mie scattering extension
cd src/physics && python setup_mie.py build_ext --inplace && cd ../..
```

---

## Configuration

All parameters are controlled via Hydra config files. Key parameters:

| Parameter | Location | Description |
|---|---|---|
| `mode` | `config.yaml` | `train`, `infer`, or `all` |
| `seed` | `config.yaml` | random seed |
| `data.spectra_per_plastic` | `data/default.yaml` | training samples per class |
| `data.batch_size` | `data/default.yaml` | batch size |
| `trainer.max_epochs` | `trainer/default.yaml` | maximum training epochs |
| `trainer.early_stopping_patience` | `trainer/default.yaml` | early stopping patience (in epochs) |
| `bg_threshold` | `config.yaml` | background probability threshold for inference filtering |
| `pred_store_path` | `config.yaml` | path to LMDB prediction store |
| `mlflow.tracking_uri` | `config.yaml` | MLflow database URI |

---

## Running

All services are run via Docker Compose. Set `PROJECT_DIR` in `.env` before starting.

### Training

```bash
docker compose run --rm train
```

To override config parameters:

```bash
docker compose run --rm train python main.py \
    mode=all \
    data.spectra_per_plastic=128 \
    seed=1
```

To run a hyperparameter sweep (Hydra multirun):

```bash
docker compose run --rm train python main.py --multirun \
    data.spectra_per_plastic=8,16,32,64,128 \
    seed="range(0,5)" \
    mode=all
```

### Inference only

```bash
docker compose run --rm train python main.py \
    mode=infer \
    ckpt_path=/app/checkpoints/best-epoch=42-val_acc=0.8500.ckpt
```

### MLflow UI

```bash
docker compose up mlflow-ui -d
# access at http://localhost:5000 (or via SSH tunnel — see below)
```

### Jupyter

```bash
docker compose up jupyter -d
# access at http://localhost:8888 (or via SSH tunnel — see below)
```

### SSH port forwarding (PuTTY)

```
Connection → SSH → Tunnels
  Source port: 5000   Destination: localhost:5000  → MLflow UI
  Source port: 8888   Destination: localhost:8888  → Jupyter
```

---


## Inference outputs

Predictions are saved to an LMDB-backed Zarr store:

```
predictions.lmdb/
    N{spectra_per_plastic}_seed{seed}/
        {image_name}/
            argmax_map      (H, W)       uint8
            bg_prob         (H, W)       float16
            top_k_classes   (H, W, k)   uint8
            top_k_probs     (H, W, k)   float16
```

Load and analyse:

```python
import zarr
import numpy as np

store   = zarr.open(zarr.LMDBStore("predictions.lmdb", readonly=True), mode="r")
grp     = store["N64_seed0/image_name"]
argmax  = grp["argmax_map"][:]
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
            acc  = float(np.mean(argmax[mask] == true_idx)) if mask.sum() > 0 else np.nan
            results.append({"N": grp.attrs["N"], "seed": grp.attrs["seed"],
                            "image": image_name, "threshold": threshold, "accuracy": acc})

df = pd.DataFrame(results)
print(df.groupby(["N", "threshold"])["accuracy"].agg(["mean", "std"]).round(4))
```

---

## Cython extension

The Mie scattering physics module must be compiled before use:

```bash
cd src/physics
python setup_mie.py build_ext --inplace
```

Inside Docker this runs automatically during `docker build`.

---

## License

MIT License — see `LICENSE` for details.

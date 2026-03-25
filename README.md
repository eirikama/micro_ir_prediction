# microplastics-predict

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
├── run.sh                            # convenience script for docker run
├── main.py                           # entry point
├── config_schema.py                  # typed dataclass config definitions
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
│   ├── Models/
│   │   ├── aacnn.py                  # AACNN LightningModule
│   │   └── blocks.py                 # attention-augmented conv blocks
│   ├── Data/
│   │   ├── data_utils.py             # SpectralDataModule, experiment split
│   │   └── aug_utils.py              # spectral augmentation
│   ├── Engine/
│   │   ├── trainer_engine.py         # training loop
│   │   ├── inference_engine.py       # multi-GPU inference workers
│   │   └── save_inference.py         # Zarr/LMDB output saving
│   └── Physics/
│       ├── mie.pyx                    # Cython Mie scattering implementation
│       └── setup_mie.py               # Cython build script
│
├── notebooks/                         # Jupyter notebooks for analysis
|     └── visualize_results.ipynb
├── docker/
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

### Without Docker (local development)

```bash
git clone https://github.com/yourname/microplastics-predict.git
cd microplastics-predict

python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt

# build Cython Mie scattering extension
cd src/Physics && python setup_mie.py build_ext --inplace && cd ../..
```

### With Docker (recommended)

```bash
# download sqlite tarball for offline build
mkdir -p docker
wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz -P docker/

# build image
docker build -t microplastics-predict .

# verify GPU access
docker run --gpus all microplastics-predict python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
"
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
| `trainer.early_stopping_patience` | `trainer/default.yaml` | early stopping patience |
| `bg_threshold` | `config.yaml` | background probability threshold for inference filtering |
| `pred_store_path` | `config.yaml` | path to LMDB prediction store |
| `mlflow.tracking_uri` | `config.yaml` | MLflow database URI |

---

## Running

### Using run.sh (recommended)

```bash
chmod +x run.sh

# single run
./run.sh mode=all data.spectra_per_plastic=64 seed=0 trainer.max_epochs=50

# hyperparameter sweep — 5 values of N × 5 seeds = 25 jobs
./run.sh --multirun \
    data.spectra_per_plastic=8,16,32,64,128 \
    seed="range(0,5)" \
    mode=all \
    trainer.max_epochs=50

# inference only with existing checkpoint
./run.sh mode=infer ckpt_path=/app/checkpoints/best-epoch=42-val_acc=0.8500.ckpt
```

### Using docker compose

```bash
# set project directory
echo 'PROJECT_DIR=/path/to/microscopy_prediction' > .env

# training
docker compose up train

# MLflow UI — access at http://localhost:5000 via SSH tunnel
docker compose up mlflow-ui -d

# Jupyter notebook — access at http://localhost:8888 via SSH tunnel
docker compose up jupyter -d
```

### SSH port forwarding (PuTTY)

```
Connection → SSH → Tunnels
  Source port: 5000   Destination: localhost:5000  → MLflow UI
  Source port: 8888   Destination: localhost:8888  → Jupyter
```

---

## Experiment tracking

All runs are tracked with MLflow. Each run logs:

- hyperparameters — N, seed, lr, batch size, epochs, bg_threshold
- per-epoch train/val loss and accuracy via Lightning MLFlowLogger
- best val accuracy, final epoch, whether early stopping fired
- per-image inference accuracy and timing
- per-class mean accuracy
- GPU info, hostname, torch version
- full Hydra config as a YAML artifact
- label encoding as a JSON artifact

Start the UI:

```bash
docker compose up mlflow-ui -d
# open http://localhost:5000
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
cd src/Physics
python setup_mie.py build_ext --inplace
```

Inside Docker this runs automatically during `docker build`.

---


## License

MIT License — see `LICENSE` for details.

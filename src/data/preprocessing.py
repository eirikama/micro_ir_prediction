"""Config-driven spectral preprocessing.

Deliberately kept separate from `augmentation.py`, because the two differ in
every respect that matters:

  augmentation                      preprocessing
  ------------------------------    -------------------------------------
  train only                        train, val AND inference
  stochastic (per-spectrum `ratio`) deterministic
  runs first (simulates artifacts)  runs after (removes them)
  stateless                         some steps carry fitted state (EMSC)

The train/inference symmetry is the important one. `AACNN` starts with
`InputNorm = InstanceNorm1d(affine=True)`, so the model is already invariant
to per-spectrum offset and scale — which is why `data.z_normalize` being
applied in the datamodule but not on the inference path is harmless. That
does NOT hold for anything here that changes spectral *shape* (derivatives,
EMSC): skip it at inference and the model sees a different domain than it
was trained on. `apply_preprocessing` is therefore called from both paths.

Steps are an ordered LIST, not a map, because order changes the result
(EMSC-then-derivative != derivative-then-EMSC) and because the same step may
legitimately appear twice.

There are two independent off switches: one per step, and one for the block
as a whole. Both config shapes below are accepted — use the second when you
want the master switch:

    preprocessing:                    preprocessing:
      - type: crop                      enabled: True        # <- master switch
        keep: [[1000, 1800]]            steps:
      - type: emsc                        - type: crop
        enabled: False   # <- per step        keep: [[1000, 1800]]
        poly_order: 2                       - type: emsc
                                              enabled: False   # <- per step
                                              poly_order: 2

`enabled` defaults to True everywhere, so omitting it keeps a step on.
Disabling the block is not the same as deleting it: a disabled step is still
validated (a typo in `type` raises rather than being silently skipped).

Stateful steps are fitted once via `fit_preprocessing(X_train, wn, cfg)`,
which returns a plain dict of numpy arrays. Persist it next to the
checkpoint (`save_state`/`load_state`) so inference reproduces training
exactly.
"""
from __future__ import annotations

import numpy as np

# The EMSC family and wavenumber interpolation come from biospectools (pinned
# in requirements.txt) — the reference implementation from the group that
# published these methods. Nothing here reimplements them: a second version
# would only drift out of sync and make results depend on which one ran.
# The import is guarded solely so that the rest of this module (crop, savgol,
# snv, vector_norm, als) still imports in an environment without it; the
# EMSC-family steps then raise with an actionable message instead of
# silently falling back to something numerically different.
try:
    from biospectools.preprocessing import EMSC as _BS_EMSC
    from biospectools.preprocessing import FringeEMSC as _BS_FringeEMSC
    from biospectools.preprocessing import MeEMSC as _BS_MeEMSC
    from biospectools.utils import interp2wns as _bs_interp2wns
    _BIOSPECTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _BS_EMSC = _BS_MeEMSC = _BS_FringeEMSC = _bs_interp2wns = None
    _BIOSPECTOOLS_AVAILABLE = False


def _require_biospectools(step: str):
    if not _BIOSPECTOOLS_AVAILABLE:
        raise ImportError(
            f"Preprocessing step '{step}' requires biospectools.\n"
            "  pip install biospectools==0.4.0\n"
            "(it is already pinned in requirements.txt, so this normally means "
            "you are outside the project environment.)"
        )

# ── individual steps ─────────────────────────────────────────────────────────
# Signature: (X, wn, cfg, state) -> (X, wn)
# wn is returned too because `crop` changes the axis.


def _crop(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Keep only the listed wavenumber windows."""

    keep = cfg.get("keep", None)
    if not keep:
        return X, wn
    m = np.zeros(len(wn), dtype=bool)
    for lo, hi in keep:
        m |= (wn >= float(lo)) & (wn <= float(hi))
    if not m.any():
        raise ValueError(f"crop kept 0 channels; keep={keep}, wn range "
                         f"{wn.min():.1f}-{wn.max():.1f}")
    return X[:, m], wn[m]


def _savgol(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Savitzky-Golay smoothing / derivative.

    deriv=0 smooths; deriv=1 removes an additive baseline; deriv=2 removes an
    additive + linear baseline.   """

    from scipy.signal import savgol_filter

    window   = int(cfg.get("window", 11))
    polyorder = int(cfg.get("polyorder", 3))
    deriv    = int(cfg.get("deriv", 0))

    window = min(window, X.shape[1] if X.shape[1] % 2 else X.shape[1] - 1)
    if window % 2 == 0:
        window += 1                      # savgol requires odd
    if polyorder >= window:
        polyorder = window - 1
    return savgol_filter(X, window, polyorder, deriv=deriv, axis=1), wn


def _snv(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Standard normal variate = per-spectrum z-score.

    Note this is very close to what the model's InstanceNorm already does,
    so it is usually a no-op in terms of accuracy. Included because it is
    conventional and because it changes what *downstream* steps see.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = np.maximum(X.std(axis=1, keepdims=True), float(cfg.get("eps", 1e-3)))
    return (X - mu) / sd, wn


def _vector_norm(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Scale each spectrum to unit L2 norm (or unit area with order=1)."""
    order = int(cfg.get("order", 2))
    n = np.linalg.norm(X, ord=order, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12), wn


def _als(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Subtract an asymmetric-least-squares baseline.

    Reuses the same estimator the fluorescence augmentation fits its basis
    with, so 'what counts as background' is consistent between the two.
    """
    from src.data.augmentation import _als_baseline

    base = _als_baseline(
        X,
        lam=float(cfg.get("lam", 1e5)),
        p=float(cfg.get("p", 0.01)),
        n_iter=int(cfg.get("n_iter", 10)),
    )
    return X - base, wn


def _reference(state: dict, X: np.ndarray, step: str) -> np.ndarray:
    """Fetch and validate the fitted reference spectrum."""
    ref = state.get("emsc_reference")
    if ref is None:
        raise RuntimeError(
            f"'{step}' requires a fitted reference. Call fit_preprocessing() on "
            "the training set and pass its state (and persist it next to the "
            "checkpoint for inference)."
        )
    if len(ref) != X.shape[1]:
        raise ValueError(
            f"{step} reference has {len(ref)} channels but spectra have "
            f"{X.shape[1]} — a `crop` step probably ran in a different order "
            "at fit time than at apply time."
        )
    return np.asarray(ref, dtype=np.float64)


def _to_plain_dict(cfg) -> dict:
    """OmegaConf node -> plain dict, so values are real Python objects."""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _as_array_if_sequence(v):
    """Sequence config values -> ndarray.

    biospectools requires arrays for `n0s`/`radiuses` (a plain list raises
    `TypeError: can't multiply sequence by non-int`), and tolerates arrays
    everywhere else a sequence is accepted, so one rule covers every param.
    """
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float64)
    return v


_EMSC_CLASSES = (
    {"emsc": _BS_EMSC, "me_emsc": _BS_MeEMSC, "fringe_emsc": _BS_FringeEMSC}
    if _BIOSPECTOOLS_AVAILABLE else {}
)

# transform() kwargs that only some of the classes accept. EMSC warns per call
# when the reference correlates poorly with a batch; that is noise during
# training, where the reference is fixed by design.
_TRANSFORM_KWARGS = {"emsc": {"check_correlation": False}}


def _emsc_family(X: np.ndarray, wn: np.ndarray, cfg, state):
    """EMSC and its extensions, delegated wholesale to biospectools.

    One adapter serves `emsc`, `me_emsc` and `fringe_emsc` because all three
    biospectools classes share a shape: `Cls(reference, wavenumbers, **params)`
    then `.transform(spectra)`. Config keys are passed straight through to the
    constructor, so biospectools' own documentation is the reference for what
    is available (interferents, analytes, weights, tol, patience, ...) and
    nothing has to be re-enumerated here when it gains a parameter.

    What this adapter does add, and why it cannot simply be dropped:

    * the registry contract `(X, wn, cfg, state) -> (X, wn)`, so steps chain;
    * the reference spectrum comes from fitted `state` — fitted once on the
      training split and reloaded at inference from the checkpoint sidecar.
      biospectools takes `reference` as a plain argument and knows nothing
      about that, and getting it wrong is precisely how train and inference
      end up correcting against different references;
    * a channel-count check, so a mis-ordered `crop` gives an actionable
      error instead of a broadcast failure deep inside biospectools.

    Which to use: `emsc` for a smooth polynomial baseline plus scaling;
    `me_emsc` when resonant Mie scattering dominates (whole microplastic
    particles, pollen grains, single cells) — note it is the physical inverse
    of the `mie_scattering` augmentation; `fringe_emsc` for thin-film samples
    where internal reflection adds a periodic ripple (needs
    `fringe_wn_location: [lo, hi]`, a signal-free window).
    """
    name = str(cfg["type"])
    _require_biospectools(name)

    params = {
        k: _as_array_if_sequence(v)
        for k, v in _to_plain_dict(cfg).items()
        if k not in ("type", "enabled")
    }
    corrected = _EMSC_CLASSES[name](
        reference=_reference(state, X, name),
        wavenumbers=np.asarray(wn, dtype=np.float64),
        **params,
    ).transform(X, **_TRANSFORM_KWARGS.get(name, {}))
    return np.asarray(corrected, dtype=np.float64), wn


def _interpolate(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Resample onto a fixed wavenumber grid (biospectools.interp2wns).

    Use when spectra from different instruments must share one axis, or to
    downsample. Because it pins the axis explicitly it also makes the model's
    input width independent of whatever the store happens to hold.

    Config keys
    -----------
    start, stop, num : the target grid, np.linspace semantics
    kind             : interpolation kind (default 'linear')
    extrapolation    : passed through; None fills out-of-range with fill_value
    """
    _require_biospectools("interpolate")
    new_wn = np.linspace(
        float(cfg["start"]), float(cfg["stop"]), int(cfg["num"])
    )
    # interp2wns returns (spectra, wavenumbers) — spectra first.
    out, out_wn = _bs_interp2wns(
        np.asarray(wn, dtype=np.float64), new_wn, X,
        kind=str(cfg.get("kind", "linear")),
        extrapolation=cfg.get("extrapolation", None),
        fill_value=float(cfg.get("fill_value", 0.0)),
    )
    return np.asarray(out, dtype=np.float64), np.asarray(out_wn, dtype=np.float64)


# Loaded DL correction models, keyed by (target, checkpoint, device).
# Module-level so a model is loaded ONCE per process rather than per batch —
# this step runs on every training batch and on every inference batch, and
# reloading a checkpoint each time is exactly the mistake that made the
# inference loop slow before (see InferenceSession).
_DL_MODEL_CACHE: dict = {}


def _resolve_device(spec):
    import torch
    spec = str(spec or "auto")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _instantiate(model_cfg):
    """Build an object from a `_target_` block.

    Uses `hydra.utils.instantiate` when Hydra is importable (so nested
    `_target_`s, interpolation and partials behave as everywhere else in this
    repo), and otherwise falls back to a plain importlib construction. The
    fallback keeps this data-layer module usable outside a Hydra app — tests,
    notebooks, `predict.py` — instead of making Hydra a hard requirement of
    preprocessing.
    """
    params = _to_plain_dict(model_cfg)
    target = params.pop("_target_", None)
    if not target:
        raise KeyError("model block needs a `_target_` naming the network class")
    try:
        from hydra.utils import instantiate
    except ImportError:
        import importlib
        module_path, _, cls_name = str(target).rpartition(".")
        cls = getattr(importlib.import_module(module_path), cls_name)
        return cls(**params)
    return instantiate(model_cfg)


def _load_dl_model(cfg, device):
    """Instantiate and cache the correction network.

    Two ways to supply it, so the network can be dropped in later without
    touching this file:

    1. `model:` block with Hydra's `_target_` (plus any constructor kwargs),
       optionally with `checkpoint:` holding a state_dict. This is the
       idiomatic route for this repo.
    2. `checkpoint:` alone pointing at a TorchScript archive
       (`torch.jit.save`), for a model exported from elsewhere.
    """
    import torch

    model_cfg  = cfg.get("model", None)
    checkpoint = cfg.get("checkpoint", None)
    target     = model_cfg.get("_target_") if model_cfg else None
    key        = (str(target), str(checkpoint), str(device))
    if key in _DL_MODEL_CACHE:
        return _DL_MODEL_CACHE[key]

    if model_cfg is not None:
        model = _instantiate(model_cfg)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                sd = sd.get("state_dict", sd)
            model.load_state_dict(sd)
    elif checkpoint:
        model = torch.jit.load(checkpoint, map_location="cpu")
    else:
        raise NotImplementedError(
            "Step 'dl_mie' has no model yet. Supply either\n"
            "  model:\n"
            "    _target_: your.module.YourMieNet\n"
            "  checkpoint: /path/to/weights.ckpt     # optional state_dict\n"
            "or a TorchScript archive:\n"
            "  checkpoint: /path/to/scripted_model.pt"
        )

    model = model.to(device).eval()
    _DL_MODEL_CACHE[key] = model
    return model


def _dl_mie(X: np.ndarray, wn: np.ndarray, cfg, state):
    """Deep-learning Mie scattering correction.

    A learned alternative to `me_emsc`: instead of iteratively fitting a Mie
    extinction basis per spectrum, a pretrained network maps a scatter-
    distorted spectrum to its corrected form in one forward pass. That makes
    it usable per training batch, where ME-EMSC's ~100 ms/spectrum is not.

    The network itself is supplied via config (see `_load_dl_model`) — this
    function only owns the plumbing: device placement, batching, caching, and
    the numpy<->torch boundary.

    Model contract
    --------------
    Input : float32 tensor `(B, 1, L)` — same layout AACNN consumes.
    Output: `(B, 1, L)` or `(B, L)`; the channel axis is squeezed if present.
    L is whatever the preceding steps produced, so if the network is fixed to
    a particular wavenumber grid put an `interpolate` step before this one.

    Config keys
    -----------
    model      : Hydra `_target_` block for the network (+ constructor kwargs)
    checkpoint : state_dict for the above, or a TorchScript archive on its own
    device     : 'auto' (default) | 'cpu' | 'cuda' | 'cuda:1'
    batch_size : spectra per forward pass (default 4096)

    Note this is the learned inverse of the `mie_scattering` augmentation —
    enabling both simulates scattering and then removes it.
    """
    import torch

    device = _resolve_device(cfg.get("device", "auto"))
    model  = _load_dl_model(cfg, device)
    bs     = int(cfg.get("batch_size", 4096))

    out = np.empty_like(X, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, X.shape[0], bs):
            chunk = torch.from_numpy(
                np.ascontiguousarray(X[start:start + bs], dtype=np.float32)
            ).unsqueeze(1).to(device)
            y = model(chunk)
            if y.ndim == 3:
                y = y.squeeze(1)
            out[start:start + bs] = y.detach().float().cpu().numpy()

    return out, wn


PREPROC_REGISTRY: dict = {
    "crop":        _crop,
    "savgol":      _savgol,
    "snv":         _snv,
    "vector_norm": _vector_norm,
    "als":         _als,
    "emsc":        _emsc_family,
    "me_emsc":     _emsc_family,
    "fringe_emsc": _emsc_family,
    "interpolate": _interpolate,
    "dl_mie":      _dl_mie,
}

# Steps needing a fit on the training set before they can be applied.
# `dl_mie` is deliberately NOT here: its network is pretrained and loaded from
# config, so train and inference stay in sync via the checkpoint path rather
# than via fitted state.
STATEFUL = {"emsc", "me_emsc", "fringe_emsc"}


# ── driver ───────────────────────────────────────────────────────────────────

def resolve_steps(block) -> list:
    """Normalise a `preprocessing:` config block into a list of enabled steps.

    Accepts either a bare list of steps, or a dict with a master `enabled`
    switch and a `steps` list. Returns [] when the block is missing, empty,
    or switched off as a whole.

    Every step's `type` is validated even when that step (or the whole block)
    is disabled, so a typo surfaces immediately instead of silently doing
    nothing the one time you switch it back on.
    """
    if not block:
        return []

    # Detect the mapping form by `.keys()`, not by isinstance or `.get`:
    # OmegaConf's ListConfig is not a `list` instance and *does* expose a
    # `.get`, so both of those tests send a plain list of steps down the
    # dict branch.
    if hasattr(block, "keys"):
        steps, block_on = list(block.get("steps") or []), bool(block.get("enabled", True))
    else:
        steps, block_on = list(block), True

    for step in steps:
        name = step.get("type")
        if name not in PREPROC_REGISTRY:
            raise KeyError(
                f"Unknown preprocessing step '{name}'. "
                f"Known: {sorted(PREPROC_REGISTRY)}"
            )

    if not block_on:
        return []
    return [s for s in steps if s.get("enabled", True)]


def apply_preprocessing(X: np.ndarray, wn: np.ndarray, block, state: dict | None = None):
    """Run the configured steps in order. Returns (X, wn)."""
    steps = resolve_steps(block)
    if not steps:
        return X, wn
    state = state or {}
    X = np.asarray(X, dtype=np.float64)
    for step in steps:
        X, wn = PREPROC_REGISTRY[step["type"]](X, wn, step, state)
    return X, wn


def fit_preprocessing(X_train: np.ndarray, wn: np.ndarray, block) -> dict:
    """Fit whatever the configured steps need, on the TRAINING set only.

    Steps are walked in order and applied as we go, so each stateful step is
    fitted on exactly the representation it will see at apply time (e.g. an
    EMSC reference after a preceding crop is on the cropped axis).
    """
    state: dict = {}
    steps = resolve_steps(block)
    if not steps:
        return state

    X = np.asarray(X_train, dtype=np.float64)
    for step in steps:
        name = step["type"]
        if name in STATEFUL and "emsc_reference" not in state:
            # Mean training spectrum. Shared by every EMSC-family step, and
            # captured at this point in the chain so it matches the
            # representation the step will actually see at apply time.
            state["emsc_reference"] = X.mean(axis=0)
        X, wn = PREPROC_REGISTRY[name](X, wn, step, state)
    return state


def save_state(state: dict, path: str) -> None:
    """Persist fitted state beside the checkpoint so inference can reuse it."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in state.items()})


def load_state(path: str) -> dict:
    if not path:
        return {}
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def sidecar_path(ckpt_path: str) -> str:
    """Where the fitted preprocessing state lives for a given checkpoint.

    Keyed to the checkpoint rather than the run directory so that inference
    (and predict.py) can recover the exact state from a checkpoint path
    alone, with no access to the training config or the training split.
    """
    from pathlib import Path
    return str(Path(ckpt_path).with_suffix("")) + ".preproc.npz"


def save_state_for_ckpt(state: dict, ckpt_path: str) -> str | None:
    """Write fitted state next to `ckpt_path`. No-op when there is none."""
    if not state or not ckpt_path:
        return None
    path = sidecar_path(ckpt_path)
    save_state(state, path)
    return path


def load_state_for_ckpt(ckpt_path: str) -> dict:
    """Load fitted state for a checkpoint, or {} when the sidecar is absent.

    Absent is legitimate — a stateless pipeline (crop/savgol only) writes no
    sidecar. A *stateful* step with no state raises later, at apply time,
    with a message pointing at the fit step.
    """
    import os
    if not ckpt_path:
        return {}
    path = sidecar_path(ckpt_path)
    return load_state(path) if os.path.exists(path) else {}

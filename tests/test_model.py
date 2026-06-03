"""Tests for the AACNN LightningModule."""
import pytest
import torch

from tests.conftest import N_CLASSES, SPECTRAL_LEN


@pytest.fixture
def batch():
    torch.manual_seed(0)
    x = torch.randn(8, 1, SPECTRAL_LEN)
    y = torch.randint(0, N_CLASSES, (8,))
    return x, y


# ── forward pass ──────────────────────────────────────────────────────────────

def test_forward_output_shape(tiny_model, batch):
    x, _ = batch
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(x)
    assert out.shape == (8, N_CLASSES)


def test_forward_no_nan(tiny_model, batch):
    x, _ = batch
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(x)
    assert torch.isfinite(out).all()


def test_forward_batch_size_one(tiny_model):
    x = torch.randn(1, 1, SPECTRAL_LEN)
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(x)
    assert out.shape == (1, N_CLASSES)


def test_forward_large_batch(tiny_model):
    x = torch.randn(64, 1, SPECTRAL_LEN)
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(x)
    assert out.shape == (64, N_CLASSES)


# ── training / validation steps ───────────────────────────────────────────────

def test_training_step_returns_scalar(tiny_model, batch):
    tiny_model.train()
    loss = tiny_model.training_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar(tiny_model, batch):
    tiny_model.eval()
    loss = tiny_model.validation_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ── parameter count ───────────────────────────────────────────────────────────

def test_model_has_trainable_parameters(tiny_model):
    n = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    assert n > 0

"""Tests for FocalLoss."""
import pytest
import torch
import torch.nn.functional as F

from src.models.loss import FocalLoss


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits  = torch.randn(32, 5)
    targets = torch.randint(0, 5, (32,))
    return logits, targets


def test_output_is_scalar(batch):
    logits, targets = batch
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.ndim == 0


def test_gamma_zero_no_alpha_equals_cross_entropy(batch):
    """With gamma=0 and no alpha, focal loss == standard cross-entropy."""
    logits, targets = batch
    fl = FocalLoss(gamma=0.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-5)


def test_with_uniform_alpha(batch):
    """Uniform alpha weights should not change relative class ordering."""
    logits, targets = batch
    n_classes = logits.shape[1]
    alpha = torch.ones(n_classes)
    loss = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_high_gamma_downweights_easy_examples():
    """High gamma should give lower loss for confidently correct predictions."""
    # Make one sample very easy (high logit on correct class)
    logits_easy = torch.tensor([[10.0, -10.0, -10.0]])
    logits_hard = torch.tensor([[1.0,  -1.0,  -1.0]])
    target = torch.tensor([0])

    loss_easy = FocalLoss(gamma=5.0)(logits_easy, target)
    loss_hard = FocalLoss(gamma=5.0)(logits_hard, target)
    assert loss_easy < loss_hard


def test_no_nan_on_random_inputs():
    torch.manual_seed(1)
    logits  = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))
    alpha   = torch.rand(10)
    loss = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)
    assert torch.isfinite(loss)

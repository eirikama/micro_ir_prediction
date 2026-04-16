import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig

from src.models.blocks import AugmentedConv, InputNorm
from src.models.loss import FocalLoss


class AACNN(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_norm = InputNorm(1)
        self.shortcut_1 = nn.Conv1d(1, config.conv_channels, 1)
        self.aconv_1 = AugmentedConv(1, config.conv_channels, config.kernel_size, dk=1, dv=1, Nh=1)
        self.bn_1 = nn.BatchNorm1d(config.conv_channels)
        self.aconv_2 = AugmentedConv(
            config.conv_channels,
            config.conv_channels,
            config.kernel_size,
            dk=16,
            dv=8,
            Nh=8,
        )
        self.bn_2 = nn.BatchNorm1d(config.conv_channels)
        self.aconv_3 = AugmentedConv(config.conv_channels, 8, config.kernel_size, dk=4, dv=4, Nh=2)
        self.bn_3 = nn.BatchNorm1d(8)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(95)

        self.fc1 = nn.Linear(760, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, config.num_classes)
        self.leaky = nn.LeakyReLU()
        self.pred_dropout = nn.Dropout(config.pred_dropout)

        alpha_weights = torch.tensor(config.alpha)
        self.loss_fn = FocalLoss(gamma=config.gamma, alpha=alpha_weights)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        acnn1 = self.leaky(self.aconv_1(x))
        x = self.leaky(self.bn_1(acnn1 + self.shortcut_1(x)))
        x = F.max_pool1d(x, 2)

        acnn2 = self.leaky(self.aconv_2(x))
        x = self.leaky(self.bn_2(acnn2 + x))
        x = F.max_pool1d(x, 2)

        acnn3 = self.leaky(self.aconv_3(x))
        x = self.bn_3(acnn3)
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)

        x = self.pred_dropout(self.leaky(self.fc1(x)))
        x = self.pred_dropout(self.leaky(self.fc2(x)))
        return self.fc3(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        spectra, y = batch
        logits = self(spectra)
        loss = self.loss_fn(logits.float(), y)
        acc = self.train_acc(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        spectra, y = batch
        logits = self(spectra)
        loss = self.loss_fn(logits.float(), y)
        acc = self.val_acc(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_acc.reset()

    def configure_optimizers(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

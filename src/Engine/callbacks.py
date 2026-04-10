import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class ExtendedLogger(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_acc = metrics.get("train_acc")
        val_acc = metrics.get("val_acc")
        if train_acc is not None and val_acc is not None:
            gap = abs(train_acc - val_acc)
            pl_module.log("Trainer/acc_overfit_gap", gap, on_epoch=True)

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        if train_loss is not None and val_loss is not None:
            gap = abs(train_loss - val_loss)
            pl_module.log("Trainer/loss_overfit_gap", gap, on_epoch=True)

        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                wait = float(getattr(cb, "wait_count", 0))
                patience = float(cb.patience)
                pl_module.log("Trainer/es_wait_count", wait, on_epoch=True)
                pl_module.log("Trainer/es_patience_used_pct", (wait / patience) * 100)

        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("Trainer/lr", current_lr, on_epoch=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        pl_module.log("Trainer/grad_norm", total_norm, on_step=True)

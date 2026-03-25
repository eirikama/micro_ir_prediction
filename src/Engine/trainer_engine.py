import logging
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore", ".*tensorboardX.*")
warnings.filterwarnings("ignore", ".*litlogger.*")
warnings.filterwarnings("ignore", ".*limit_train_batches.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty.*")
warnings.filterwarnings("ignore", ".*Precision 16-mixed is not supported by the model summary.*")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run_training(cfg, model, datamodule, logger=None):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=cfg.trainer.early_stopping_patience,
        verbose=False,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop],
        limit_train_batches=datamodule.steps_per_epoch,
        limit_val_batches=10,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

    score = trainer.checkpoint_callback.best_model_score
    best_score = float(score) if score is not None else 0.0

    early_stop_cb = next((cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)), None)
    stopped_early = early_stop_cb is not None and trainer.current_epoch < cfg.trainer.max_epochs - 1

    return (
        trainer.checkpoint_callback.best_model_path,
        best_score,  # ← comma fixed
        trainer.current_epoch,
        stopped_early,
    )

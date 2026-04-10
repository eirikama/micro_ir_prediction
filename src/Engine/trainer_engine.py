import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.Engine.callbacks import ExtendedLogger


def run_training(cfg, model, datamodule, logger=None):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    patience = cfg.trainer.early_stopping_patience // cfg.trainer.val_every_n_epochs

    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=max(5, patience),
        verbose=False,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, early_stop, ExtendedLogger()],
        limit_train_batches=datamodule.steps_per_epoch,
        limit_val_batches=datamodule.val_batches,
        log_every_n_steps=datamodule.steps_per_epoch,
        check_val_every_n_epoch=cfg.trainer.val_every_n_epochs,
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
        best_score,
        trainer.current_epoch,
        stopped_early,
    )

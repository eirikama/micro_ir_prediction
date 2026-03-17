import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def run_training(cfg, model, datamodule):
    """Encapsulates the Trainer setup and execution."""
    
    # Define callbacks locally so they are fresh for every run
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-{epoch:02d}",
        save_top_k=1,
        mode="min"
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stopping_patience
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
    )

    trainer.fit(model, datamodule=datamodule)
    
    return checkpoint_callback.best_model_path
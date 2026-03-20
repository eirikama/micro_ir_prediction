import warnings
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from tqdm import tqdm


warnings.filterwarnings("ignore", ".*tensorboardX.*")
warnings.filterwarnings("ignore", ".*litlogger.*")
warnings.filterwarnings("ignore", ".*limit_train_batches.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty.*")
warnings.filterwarnings("ignore", ".*Precision 16-mixed is not supported by the model summary.*")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class EpochBar(TQDMProgressBar):
    def init_train_tqdm(self):
        # don't reference self.trainer here — it may not be set yet
        return tqdm(
            desc          = "Training",
            total         = 0,           # set correctly in on_train_start
            dynamic_ncols = True,
            leave         = True,
            position      = 0,
        )

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # set total now that trainer is available
        self.train_progress_bar.total  = trainer.max_epochs
        self.train_progress_bar.n      = 0
        self.train_progress_bar.refresh()

    def init_validation_tqdm(self):
        return tqdm(disable=True)

    def init_sanity_tqdm(self):
        return tqdm(disable=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass   

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass  

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        self.train_progress_bar.set_postfix(metrics)
        self.train_progress_bar.update(1)
        self.train_progress_bar.refresh()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        self.train_progress_bar.set_postfix(metrics)
        self.train_progress_bar.refresh()

    def on_train_end(self, trainer, pl_module):
        pass   

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics

def run_training(cfg, model, datamodule, logger=None):

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    early_stop = EarlyStopping(
        monitor="val_acc", patience=cfg.trainer.early_stopping_patience
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop, EpochBar(refresh_rate=1)],
        limit_train_batches=datamodule.steps_per_epoch,
        limit_val_batches=10,
        log_every_n_steps=10,
        enable_model_summary=False,
        num_sanity_val_steps = 0,
        logger=logger
    )

    trainer.fit(model, datamodule=datamodule)

    return checkpoint_callback.best_model_path

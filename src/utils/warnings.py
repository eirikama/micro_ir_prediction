import logging
import warnings


def silence_warnings():
    warnings.filterwarnings("ignore", ".*tensorboardX.*")
    warnings.filterwarnings("ignore", ".*litlogger.*")
    warnings.filterwarnings("ignore", ".*limit_train_batches.*")
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty.*")
    warnings.filterwarnings(
        "ignore", ".*Precision 16-mixed is not supported by the model summary.*"
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

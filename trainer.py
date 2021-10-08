# -*- coding: utf-8 -*-
r"""
Lightning Trainer Setup
=======================
   Setup logic for the lightning trainer.
"""
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from utils import Config


class TrainerConfig(Config):
    """
    The TrainerConfig class is used to define default hyper-parameters that
    are used to initialize our Lightning Trainer. These parameters are then overwritted
    with the values defined in the YAML file.
    -------------------- General Parameters -------------------------
    :param seed: Training seed.
    :param deterministic: If true enables cudnn.deterministic. Might make your system
        slower, but ensures reproducibility.
    :param verbose: verbosity mode.
    :param overfit_batches: Uses this much data of the training set. If nonzero, will use
        the same training set for validation and testing. If the training dataloaders
        have shuffle=True, Lightning will automatically disable it.
    -------------------- Model Checkpoint & Early Stopping -------------------------
    :param early_stopping: If true enables EarlyStopping.
    :param save_top_k: If save_top_k == k, the best k models according to the metric
        monitored will be saved.
    :param monitor: Metric to be monitored.
    :param save_weights_only: Saves only the weights of the model.
    :param metric_mode: One of {min, max}. In min mode, training will stop when the
        metric monitored has stopped decreasing; in max mode it will stop when the
        metric monitored has stopped increasing.
    :param min_delta: Minimum change in the monitored metric to qualify as an improvement.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param accumulate_grad_batches: Gradient accumulation steps.
    """

    seed: int = 3
    deterministic: bool = True
    verbose: bool = False
    overfit_batches: float = 0.0

    # Model Checkpoint & Early Stopping
    early_stopping: bool = True
    save_top_k: int = 1
    monitor: str = "kendall"
    save_weights_only: bool = False
    metric_mode: str = "max"
    min_delta: float = 0.0
    patience: int = 1
    accumulate_grad_batches: int = 1

    def __init__(self, initial_data: dict) -> None:
        trainer_attr = pl.Trainer.default_attributes()
        for key in trainer_attr:
            setattr(self, key, trainer_attr[key])

        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])


def build_trainer(hparams: Namespace) -> pl.Trainer:
    """
    :param hparams: Namespace
    Returns:
        - pytorch_lightning Trainer
    """
    # Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=hparams.verbose,
        mode=hparams.metric_mode,
    )

    # TestTube Logger Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="lightning",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join("experiments/lightning/", tb_logger.version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=hparams.verbose,
        monitor=hparams.monitor,
        save_weights_only=hparams.save_weights_only,
        period=1,
        mode=hparams.metric_mode,
    )
    print('Logging to: ', ckpt_path)

    other_callbacks = [early_stop_callback, checkpoint_callback, LearningRateMonitor()]
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=other_callbacks,
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        log_gpu_memory=None,
        deterministic=hparams.deterministic,
        overfit_batches=hparams.overfit_batches,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        val_check_interval=hparams.val_check_interval,
        distributed_backend=hparams.distributed_backend,
        precision=hparams.precision,
        weights_summary="top",
        profiler=hparams.profiler,
    )
    return trainer
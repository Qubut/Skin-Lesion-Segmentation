"""
Skin Lesion Segmentation Training Script with Hyperparameter Optimization
=======================================================================

This script provides three modes of operation:
1. Normal training with fixed parameters
2. Hyperparameter optimization using Optuna
3. Training with best parameters from optimization

Usage Examples:
--------------
1. Normal training:
   python train.py data=img_size=256 training.batch_size=32 training.augmentation=True

2. Hyperparameter optimization:
   python train.py optimize=True optuna.n_trials=50

3. Training with best parameters (after optimization):
   python train.py data.img_size=256 training.batch_size=64 training.augmentation=True
   model.learning_rate=0.000316
"""

import os
import sys
import optuna
import hydra
from hydra.utils import instantiate, to_absolute_path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from loguru import logger as loguruLogger
from omegaconf import OmegaConf
import sys
sys.path.extend(['/content/Skin-Lesion-Segmentation/skin_lesion_segmentation/models/', '/content/Skin-Lesion-Segmentation/skin_lesion_segmentation/datasets/'])
from skin_lesion_segmentation.models.lesion_segmentation_module import LesionSegmentationModule
from skin_lesion_segmentation.models.callbacks import OptunaPruning
from skin_lesion_segmentation.datasets.lesion_dataset_with_augementation import LesionDataModule

# Configure logging
loguruLogger.remove()
loguruLogger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
loguruLogger.add("logs/training.log", rotation="5 MB", level="DEBUG")

@hydra.main(config_path="./configs/", config_name="config.yaml", version_base="1.1")
def main(cfg):
    """
    Main training function with Optuna integration

    Args:
        cfg: Hydra configuration object

    Modes of Operation:
    1. Normal training when cfg.optimize=False
    2. Hyperparameter search when cfg.optimize=True
    """
    loguruLogger.info("Starting execution with configuration:")
    loguruLogger.debug(OmegaConf.to_yaml(cfg))
    data_path = to_absolute_path(cfg.data.data_path)
    cfg.data.data_path = data_path
    if cfg.optimize:
        run_hyperparameter_search(cfg)
    else:
        run_normal_training(cfg)

def run_normal_training(cfg):
    """
    Standard training procedure with fixed parameters

    Args:
        cfg: Configuration containing all parameters
    """
    loguruLogger.info("Initializing training components")
    data_module = instantiate(cfg.data_module)
    model = LesionSegmentationModule(**cfg.model)

    loguruLogger.info(f"Creating trainer with max_epochs={cfg.training.max_epochs}")
    checkpoint_callback = instantiate(cfg.checkpoint)
    early_stopping_callback = instantiate(cfg.early_stopping)
    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(
        **cfg.training,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger
    )

    loguruLogger.info("Starting training process")
    trainer.fit(model, datamodule=data_module)

    loguruLogger.info("Starting testing process")
    trainer.test(model, datamodule=data_module)

def run_hyperparameter_search(cfg):
    """
    Optuna-based hyperparameter optimization

    Args:
        cfg: Base configuration
    """
    loguruLogger.warning("Starting hyperparameter optimization mode")

    study = optuna.create_study(
        direction='maximize',
        storage=cfg.optuna.storage,
        study_name=cfg.optuna.study_name,
        load_if_exists=True
    )

    try:
        study.optimize(lambda trial: objective(trial, cfg),
                      n_trials=cfg.optuna.n_trials,
                      timeout=cfg.optuna.timeout)
    except Exception as e:
        loguruLogger.exception("Optimization failed", exc_info=e)
        raise

    loguruLogger.success(f"Optimization completed with best value: {study.best_value}")
    loguruLogger.success(f"Best parameters: {study.best_params}")

def objective(trial, base_cfg):
    """
    Optuna objective function for hyperparameter optimization

    Args:
        trial: Optuna trial object
        base_cfg: Base configuration from Hydra

    Returns:
        float: Best validation DICE score for this trial
    """
    loguruLogger.debug(f"Starting trial {trial.number}")

    # Define hyperparameter search space
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    img_size = trial.suggest_categorical('img_size', [128, 256, 512])
    augmentation = trial.suggest_categorical('augmentation', [True, False])

    trial_cfg = {
        "data_module": {
            "img_size": img_size,
            "batch_size": batch_size,
            "augmentation": augmentation,
            "data_path": base_cfg.data.data_path,
            "train_split": base_cfg.data.train_split
        },
        "model": {
            "num_classes": base_cfg.model.num_classes,
            "learning_rate": lr
        },
        "trainer": {
            "max_epochs": base_cfg.training.max_epochs,
            "accelerator": base_cfg.training.accelerator,
            "devices": 1
        },
        "checkpoint": {
            "monitor": "val_dice",
            "dirpath": f"checkpoints/trial_{trial.number}",
            "filename": "best",
            "save_top_k": 1,
            "mode": "max"
        },
        "early_stopping": {
            "monitor": "val_dice",
            "patience": 15,
            "mode": "max"
        },
        "logger": {
            "save_dir": f"logs/optuna/trial_{trial.number}",
            "name": "",
            "version": ""
        }
    }

    try:
        data_module = LesionDataModule(**trial_cfg['data_module'])
        data_module.setup()
        model = LesionSegmentationModule(**trial_cfg['model'])
        logger = CSVLogger(**trial_cfg['logger'])

        trainer = pl.Trainer(
            **trial_cfg['trainer'],
            callbacks=[
                ModelCheckpoint(**trial_cfg['checkpoint']),
                EarlyStopping(**trial_cfg['early_stopping']),
                OptunaPruning(trial, monitor="val_dice")
            ],
            logger=logger
        )

        trainer.fit(model, datamodule=data_module)

        metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
        best_epoch = metrics['val_dice'].idxmax()
        return metrics.loc[best_epoch, 'val_dice']

    except Exception as e:
        loguruLogger.error(f"Trial {trial.number} failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        loguruLogger.exception("Fatal error occurred", exc_info=e)
        sys.exit(1)

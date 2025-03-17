import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode
from pytorch_lightning import LightningDataModule
from lesion_dataset_with_augementation import LesionDataset

class LesionDataModule(LightningDataModule):
    """
    Data module for skin lesion segmentation

    Args:
        data_path (str): Root path containing train/test directories
        train_split (float): Fraction of data for training (default: 0.8)
        img_size (int): Target image size (resized to square)
        batch_size (int): Batch size for data loaders
        augmentation (bool): Enable data augmentation (default: False)
        num_workers (int): Number of DataLoader workers (default: 4)
    """
    def __init__(self,
                 data_path: str,
                 train_split: float = 0.8,
                 img_size: int = 256,
                 batch_size: int = 32,
                 augmentation: bool = False,
                 num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.train_split = train_split
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Create datasets for training/validation/test"""
        if stage in (None, 'fit'):
            full_dataset = LesionDataset(
                root_path=self.data_path,
                img_size=self.img_size,
                augmentation=self.augmentation
            )

            # Split into train/val
            dataset_size = len(full_dataset)
            train_size = int(dataset_size * self.train_split)
            val_size = dataset_size - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        if stage in (None, 'test'):
            self.test_dataset = LesionDataset(
                root_path=self.data_path,
                img_size=self.img_size,
                test=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

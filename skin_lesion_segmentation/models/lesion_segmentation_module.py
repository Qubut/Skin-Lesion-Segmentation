import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.segmentation import DiceScore
from models.unet_modified import UNetModified

class LesionSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for skin lesion segmentation

    Args:
        num_classes (int): Number of output classes (default: 1 for binary segmentation)
        learning_rate (float): Initial learning rate
        model (nn.Module): Segmentation model (default: UNetModified)
    """
    def __init__(self,
                 num_classes: int = 1,
                 learning_rate: float = 1e-3,
                 model: nn.Module = None):
        super().__init__()
        self.save_hyperparameters()

        # Use provided model or default UNetModified
        self.model = model or UNetModified(
            in_channels=3,
            num_classes=num_classes
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_dice = DiceScore(num_classes, average='micro')
        self.val_dice = DiceScore(num_classes, average='micro')
        self.test_dice = DiceScore(num_classes, average='micro')

    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

    def _shared_step(self, batch):
        """Common steps for training/validation/testing"""
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.sigmoid(logits)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        """Training step with loss and metrics logging"""
        loss, preds, masks = self._shared_step(batch)
        dice = self.train_dice(preds, masks)

        self.log_dict({
            'train_loss': loss,
            'train_dice': dice
        }, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with metrics logging"""
        loss, preds, masks = self._shared_step(batch)
        dice = self.val_dice(preds, masks)

        self.log_dict({
            'val_loss': loss,
            'val_dice': dice
        }, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step with metrics logging"""
        loss, preds, masks = self._shared_step(batch)
        dice = self.test_dice(preds, masks)

        self.log_dict({
            'test_loss': loss,
            'test_dice': dice
        }, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'interval': 'epoch'
            }
        }

    def on_train_epoch_end(self):
        """Synchronize metrics at the end of training epoch"""
        self.train_dice.reset()

    def on_validation_epoch_end(self):
        """Synchronize metrics at the end of validation epoch"""
        self.val_dice.reset()

    def on_test_epoch_end(self):
        """Synchronize metrics at the end of test epoch"""
        self.test_dice.reset()

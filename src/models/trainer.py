"""
Training and evaluation pipeline for LunarIceNet.

Handles:
    - Training loop with physics-informed loss
    - Validation with multiple metrics
    - Checkpoint management
    - Learning rate scheduling
    - Logging and visualization of training progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
import time
import json

from src.models.lunaricenet import LunarIceNet, PhysicsInformedLoss

logger = logging.getLogger(__name__)


class Metrics:
    """Compute classification and regression metrics for ice detection."""

    @staticmethod
    def compute(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Args:
            predictions: (B, H, W) sigmoid probabilities
            targets: (B, H, W) binary labels

        Returns:
            Dict of metrics
        """
        pred_binary = (predictions > threshold).float()
        targets_binary = (targets > threshold).float()

        tp = (pred_binary * targets_binary).sum().item()
        fp = (pred_binary * (1 - targets_binary)).sum().item()
        fn = ((1 - pred_binary) * targets_binary).sum().item()
        tn = ((1 - pred_binary) * (1 - targets_binary)).sum().item()

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # IoU for ice class
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
        }


class LunarIceNetTrainer:
    """Training manager for LunarIceNet."""

    def __init__(
        self,
        model: LunarIceNet,
        config: dict,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.config = config
        train_cfg = config['training']

        self.criterion = PhysicsInformedLoss(
            bce_weight=train_cfg['loss']['components']['bce_weight'],
            depth_weight=train_cfg['loss']['components']['depth_mse_weight'],
            physics_weight=train_cfg['loss']['components']['physics_constraint'],
            temp_prior_weight=train_cfg['loss']['components']['temperature_prior'],
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

        self.epochs = train_cfg['epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=train_cfg['learning_rate'] * 0.01,
        )

        self.best_f1 = 0.0
        self.history = {'train': [], 'val': []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'bce': 0, 'depth': 0, 'physics': 0, 'temp_prior': 0}
        all_preds, all_targets = [], []
        n_batches = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            physical = batch['physical'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features, physical)
            losses = self.criterion(outputs, labels, physical)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses['total'].item()
            for k in loss_components:
                loss_components[k] += losses[k].item()
            n_batches += 1

            with torch.no_grad():
                preds = torch.sigmoid(outputs['ice_prob'].squeeze(1))
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = Metrics.compute(all_preds, all_targets)

        return {'loss': avg_loss, **avg_components, **metrics}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        n_batches = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            physical = batch['physical'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(features, physical)
            losses = self.criterion(outputs, labels, physical)

            total_loss += losses['total'].item()
            n_batches += 1

            preds = torch.sigmoid(outputs['ice_prob'].squeeze(1))
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = Metrics.compute(all_preds, all_targets)

        return {'loss': avg_loss, **metrics}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ):
        """Full training loop."""
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training LunarIceNet on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            # Logging
            logger.info(
                f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f}"
            )

            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_checkpoint(ckpt_dir / "best_model.pth", epoch, val_metrics)
                logger.info(f"  → New best F1: {self.best_f1:.4f}")

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", epoch, val_metrics)

        # Save training history
        with open(ckpt_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training complete. Best Val F1: {self.best_f1:.4f}")

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'metrics': metrics,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

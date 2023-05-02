from typing import List, Any
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric
from torch.functional import F
import torch.nn as nn
from models.models import WindNet

class WindNetPL(pl.LightningModule):

    def __init__(self,
                 cfg,
                 net: torch.nn.Module,
                 optimizer,
                 scheduler,
                 ):
        
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sigmoid = nn.Sigmoid()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_auroc_best = MaxMetric()

        self.train_auroc = torchmetrics.AUROC(num_classes=1)
        self.val_auroc = torchmetrics.AUROC(num_classes=1)
        self.test_auroc = torchmetrics.AUROC(num_classes=1)

        self.train_precision = torchmetrics.Precision(num_classes=1, threshold=cfg.threshold)
        self.val_precision = torchmetrics.Precision(num_classes=1, threshold=cfg.threshold)
        self.test_precision = torchmetrics.Precision(num_classes=1, threshold=cfg.threshold)

        self.train_recall = torchmetrics.Recall(num_classes=1, threshold=cfg.threshold)
        self.val_recall = torchmetrics.Recall(num_classes=1, threshold=cfg.threshold)
        self.test_recall = torchmetrics.Recall(num_classes=1, threshold=cfg.threshold)

        self.train_ap = torchmetrics.AveragePrecision(num_classes=1)
        self.val_ap = torchmetrics.AveragePrecision(num_classes=1)
        self.test_ap = torchmetrics.AveragePrecision(num_classes=1)

        #self.criterion = FocalLoss(gamma=5, alpha=6)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        # return self.criterion(torch.squeeze(y_hat), y)
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_auroc_best.reset()

    def model_step(self, batch):
        objs, target = batch
        target = torch.unsqueeze(target, dim=-1)
        predictions = self(objs).float()
        loss = self.loss(predictions, target.float())

        return loss, self.sigmoid(predictions), target

    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.train_loss(loss)
        self.train_recall(predictions, target)
        self.train_precision(predictions, target)
        self.train_auroc(predictions, target)
        self.train_ap(predictions, target)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/AP", self.train_ap, on_step=True, on_epoch=True, prog_bar=True)


        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.val_loss(loss)
        self.val_recall(predictions, target)
        self.val_precision(predictions, target)
        self.val_auroc(predictions, target)
        self.val_ap(predictions, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/AP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        auroc = self.val_auroc.compute()
        self.val_auroc_best(auroc)
        self.log("val/auroc_best", self.val_auroc_best.compute(), prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_recall(predictions, target)
        self.test_precision(predictions, target)
        self.test_auroc(predictions, target)
        self.test_ap(predictions, target)

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True)
        self.log("test/AP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)
        
        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def configure_optimizers(self):
        lr = self.cfg.learning_rate
        optimizer = self.optimizer(self.net.parameters(), lr=lr,  weight_decay=0.03)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "patience": 2,
                    "mode": "min",
                    "factor": 0.5,
                    "verbose": True,
                    "min_lr": 1e-8,
                    'frequency': 1
                },
            }
        return {"optimizer": optimizer}


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):

        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
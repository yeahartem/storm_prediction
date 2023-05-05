from typing import List, Any
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torch.functional import F
import torch.nn as nn
from models.models import WindNet
import wandb
import numpy as np

class WindNetPL(pl.LightningModule):

    def __init__(self,
                 cfg,
                 net: torch.nn.Module,
                 optimizer,
                 scheduler,
                 criterion,
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
        self.val_MAE_best = MinMetric()

        self.train_MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.val_MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.test_MAPE = torchmetrics.MeanAbsolutePercentageError()

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        
        self.criterion = criterion

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):        
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_MAE_best.reset()

    def model_step(self, batch):
        objs, target = batch
        target = torch.unsqueeze(target, dim=-1)
        predictions = self(objs).float()
        loss = self.loss(predictions, target.float())

        return loss, predictions, target

    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)
        self.train_loss(loss)
        self.train_MAPE(predictions, target)
        self.train_MAE(predictions, target)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/MAPE", self.train_MAPE, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/MAE", self.train_MAE, on_step=True, on_epoch=True, prog_bar=True)

        batch = torch.cat(tuple(batch[0]), dim=0)
        nan_indicator = torch.isnan(batch).bool().int().sum()
        if nan_indicator > 0:
            print(nan_indicator)
        wandb.log({"train/target": target, "train/prediction": predictions})
        # wandb.log({"train/data_mean": batch.mean()})
        # wandb.log({"train/data_std": batch.std()})
        # wandb.log({"train/data_nans": nan_indicator})

        
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
        self.val_MAPE(predictions, target)
        self.val_MAE(predictions, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/MAPE", self.val_MAPE, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/MAE", self.val_MAE, on_step=False, on_epoch=True, prog_bar=True)

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        MAPE = self.val_MAE.compute()
        self.val_MAE_best(MAPE)
        self.log("val/MAE_best", self.val_MAE_best.compute(), prog_bar=False)


    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_MAPE(predictions, target)
        self.test_MAE(predictions, target)

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/MAPE", self.test_MAPE, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/MAE", self.test_MAE, on_step=False, on_epoch=True, prog_bar=True)
        
        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(),
                                   lr=self.cfg.learning_rate,
                                   weight_decay=self.cfg.weight_decay)
        
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
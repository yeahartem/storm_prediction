from typing import List, Any
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric
from torch.functional import F
import torch.nn as nn

class WindNet(nn.Module):
    def __init__(self) -> None:
        super(WindNet, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=8,
            out_channels=32,
            kernel_size=(28, 3, 3),
            stride=1,
            dilation=1,
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=16,
            kernel_size=(5, 3, 3),
            stride=1,
            dilation=1,
            padding=(1, 1, 1,)
        )
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3, 3),
            stride=1,
            dilation=1,
            padding=(1, 1, 1,)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 1)

        self.net = nn.Sequential(
            self.conv1, 
            nn.ReLU(),
            nn.InstanceNorm3d(32),
            self.conv2, 
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            self.conv3,
            nn.ReLU(),
            nn.InstanceNorm3d(16),
            self.flatten,
            self.fc
        )

    def forward(self, X) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.net(X)
        return output


class WindNetPL(pl.LightningModule):

    def __init__(self,
                 args,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.sigmoid = nn.Sigmoid()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_auroc_best = MaxMetric()

        self.train_accuracy = torchmetrics.Accuracy(num_classes=1, threshold=args["threshold"])
        self.val_accuracy = torchmetrics.Accuracy(num_classes=1, threshold=args["threshold"])
        self.test_accuracy = torchmetrics.Accuracy(num_classes=1, threshold=args["threshold"])

        self.train_auroc = torchmetrics.AUROC(num_classes=1)
        self.val_auroc = torchmetrics.AUROC(num_classes=1)
        self.test_auroc = torchmetrics.AUROC(num_classes=1)

        self.train_precision = torchmetrics.Precision(num_classes=1, threshold=args["threshold"])
        self.val_precision = torchmetrics.Precision(num_classes=1, threshold=args["threshold"])
        self.test_precision = torchmetrics.Precision(num_classes=1, threshold=args["threshold"])

        self.train_recall = torchmetrics.Recall(num_classes=1, threshold=args["threshold"])
        self.val_recall = torchmetrics.Recall(num_classes=1, threshold=args["threshold"])
        self.test_recall = torchmetrics.Recall(num_classes=1, threshold=args["threshold"])

        #self.criterion = FocalLoss(gamma=5, alpha=6)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_auroc_best.reset()

    def model_step(self, batch):
        objs, target = batch
        # target = torch.unsqueeze(target, dim=-1)
        predictions = self(objs).float()
        loss = self.loss(predictions, target.float())

        return loss, self.sigmoid(predictions), target

    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.train_loss(loss)
        self.train_accuracy(predictions, target)
        self.train_recall(predictions, target)
        self.train_precision(predictions, target)
        self.train_auroc(predictions, target)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

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
        self.val_accuracy(predictions, target)
        self.val_recall(predictions, target)
        self.val_precision(predictions, target)
        self.val_auroc(predictions, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

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
        self.test_accuracy(predictions, target)
        self.test_recall(predictions, target)
        self.test_precision(predictions, target)
        self.test_auroc(predictions, target)

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True)

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def configure_optimizers(self):
        lr = self.args["lr"]
        optimizer = self.hparams.optimizer(self.net.parameters(), lr=lr,  weight_decay=0.03)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
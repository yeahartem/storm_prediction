import sys,os
sys.path.append(os.getcwd())
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
from src.utils.metrics import float_to_binary, float_to_score, get_outliers_s, get_outliers_p

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
        self.criterion = criterion

        self.sigmoid = nn.Sigmoid()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_MAE_best = MinMetric()

        self.train_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.val_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.test_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        
        self.train_MAE_OS = torchmetrics.MeanAbsoluteError() # MAE outliers based on station measure
        self.val_MAE_OS = torchmetrics.MeanAbsoluteError() 
        self.test_MAE_OS = torchmetrics.MeanAbsoluteError() 
        self.train_MAE_OP = torchmetrics.MeanAbsoluteError() # MAE outliers based on prediction
        self.val_MAE_OP = torchmetrics.MeanAbsoluteError()
        self.test_MAE_OP = torchmetrics.MeanAbsoluteError()

        self.cfg.target_threshold = self.cfg.target_threshold/self.cfg.target_max


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
        self.train_MAE(predictions, target)
        self.train_MAE_OS(*get_outliers_s(predictions, target , thresh=self.cfg.target_threshold ))
        self.train_MAE_OP(*get_outliers_p(predictions , target , thresh=self.cfg.target_threshold ))
        self.train_AP(float_to_score(predictions , thresh=self.cfg.target_threshold ),
                       float_to_binary(target , thresh=self.cfg.target_threshold))

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/MAE", self.train_MAE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/MAE_OS", self.train_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/MAE_OP", self.train_MAE_OP, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/AP", self.train_AP, on_step=True, on_epoch=True, prog_bar=True)
        self.logger.experiment.log({"train/target": target, "train/prediction": predictions})

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
        self.val_MAE(predictions, target)
        self.val_AP(float_to_score(predictions, thresh=self.cfg.target_threshold),
                     float_to_binary(target, thresh=self.cfg.target_threshold))
        self.val_MAE_OS(*get_outliers_s(predictions, target, thresh=self.cfg.target_threshold))
        self.val_MAE_OP(*get_outliers_p(predictions, target, thresh=self.cfg.target_threshold))

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/MAE", self.val_MAE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/MAE_OS", self.val_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/MAE_OP", self.val_MAE_OP, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/AP", self.val_AP, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.log({"val/target": target, "val/prediction": predictions})

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def on_validation_epoch_end(self):

        MAPE = self.val_MAE.compute()
        self.val_MAE_best(MAPE)
        self.log("val/MAE_best", self.val_MAE_best.compute(), prog_bar=False)


    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_MAE(predictions, target)
        self.test_MAE_OS(*get_outliers_s(predictions, target, thresh=self.cfg.target_threshold))
        self.test_MAE_OP(*get_outliers_p(predictions, target, thresh=self.cfg.target_threshold))
        self.test_AP(float_to_score(predictions, thresh=self.cfg.target_threshold),
                      float_to_binary(target, thresh=self.cfg.target_threshold))

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/MAE", self.test_MAE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/MAE_OS", self.test_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/MAE_OP", self.test_MAE_OP, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/AP", self.test_AP, on_step=False, on_epoch=True, prog_bar=True)
        
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
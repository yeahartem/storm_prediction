from torch import nn
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics

from sklearn import metrics


class WindNet(nn.Module):
    def __init__(self, args) -> None:
        super(WindNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=args["in_channels"],
            out_channels=args["out_channels_1"],  # 12
            kernel_size=args["k_size_1"],
            stride=args["stride_1"],
            dilation=args["dilation_1"],
            padding=args["k_size_1"] - 1,
        )
        self.conv2 = nn.Conv2d(  # 12, 6, 3
            in_channels=args["out_channels_1"],
            out_channels=args["out_channels_2"],
            kernel_size=args["k_size_2"],
            stride=args["stride_2"],
            dilation=args["dilation_2"],
            padding=args["k_size_2"] - 1,
        )
        self.maxpool = nn.MaxPool2d(args["maxpool_2"])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(args["fc_size"], 2)
        self.args = args
        self.net = nn.Sequential(
            self.conv1, 
            nn.ReLU(), 
            self.conv2, 
            nn.ReLU(), 
            self.maxpool, 
            self.flatten, 
            self.fc, 
            nn.LogSoftmax(), 
        ).double()

    def forward(self, X) -> torch.Tensor:
        # print(self.flatten(self.maxpool(self.conv2(self.conv1(X)))).shape)
        output = self.net(X)
        return output


class WindNetPL(pl.LightningModule):
    ## Initialize. Define latent dim, learning rate, and Adam betas
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.net = WindNet(self.args)

        self.accuracy = torchmetrics.Accuracy(
            num_classes=2, threshold=args["threshold"]
        )
        self.AUROC = torchmetrics.AUROC(num_classes=2)
        self.precision_m = torchmetrics.Precision(
            num_classes=2, threshold=args["threshold"]
        )
        self.recall = torchmetrics.Recall(num_classes=2, threshold=args["threshold"])
        self.F1 = torchmetrics.F1Score(num_classes=2, threshold=args["threshold"])
        self.conf_matrix = torchmetrics.ConfusionMatrix(
            num_classes=2, threshold=args["threshold"]
        )
        self.stats_scores = torchmetrics.StatScores(
            num_classes=2, threshold=args["threshold"]
        )

        self.loss_f = nn.NLLLoss(
            weight=torch.tensor([1.0, self.args["pos_weight"]], dtype=torch.float64)
        )  # nn.BCELoss(weight=torch.tensor([1.0, self.args["pos_weight"]]))

    def forward(self, X):
        return self.net(X)

    def loss(
        self, y_hat, y
    ):  # POS WEIGHT CLASS !!! https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # return self.loss_f(torch.log(y_hat), y)
        return self.loss_f(y_hat, y)

    def training_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs) 
        loss = self.loss(predictions, target)

        # logging

        self.log("train_loss", loss, prog_bar=True)

        tqdm_dict = {
            "train_loss": loss,
        }
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output
    # def training_epoch_end(self, training_step_outputs):
    #     auroc = self.AUROC.compute()
    #     print("AUROC = ", auroc)
            
    def training_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        conf_m = self.conf_matrix(predictions, target)
        tp, fp, fn, tn = conf_m[0, 0], conf_m[0, 1], conf_m[1, 0], conf_m[1, 1]
        acc = (tp + tn) / (conf_m.sum())
        rec = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        prec = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        f1 = (
            tp / (tp + 0.5 * (fp + fn))
            if ((tp + 0.5 * (fp + fn))) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )

        auroc = self.AUROC.compute()
        self.AUROC.update(predictions, target)

        self.logger.experiment.add_scalars(
            "clf_metrics_train",
            {
                "train_acc": acc,
                "train_recall": rec,
                "train_auroc": auroc,
                "train_prec": prec,
                "train_f1": f1,
            },
            global_step=self.global_step,
        )

    def validation_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        self.log("val_loss", loss, prog_bar=True)
        tqdm_dict = {"val_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        conf_m = self.conf_matrix(predictions, target)
        tp, fp, fn, tn = conf_m[0, 0], conf_m[0, 1], conf_m[1, 0], conf_m[1, 1]
        acc = (tp + fp) / (conf_m.sum())
        rec = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        prec = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        f1 = (
            tp / (tp + 0.5 * (fp + fn))
            if ((tp + 0.5 * (fp + fn))) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )

        
        self.AUROC.update(predictions, target)
        auroc = self.AUROC.compute()

        self.logger.experiment.add_scalars(
            "clf_metrics_val",
            {
                "val_acc": acc,
                "val_recall": rec,
                "val_auroc": auroc,
                "val_prec": prec,
                "train_f1": f1,
            },
            global_step=self.global_step,
        )
        self.log("val_auroc", auroc)

    def test_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        self.log("test_loss", loss, prog_bar=True)
        tqdm_dict = {"test_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def test_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        conf_m = self.conf_matrix(predictions, target)
        tp, fp, fn, tn = conf_m[0, 0], conf_m[0, 1], conf_m[1, 0], conf_m[1, 1]
        acc = (tp + fp) / (conf_m.sum())
        rec = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        prec = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        f1 = (
            tp / (tp + 0.5 * (fp + fn))
            if ((tp + 0.5 * (fp + fn))) > 0
            else torch.tensor(0.0, dtype=target.dtype, device=target.device)
        )
        self.AUROC.update(predictions, target)
        auroc = self.AUROC.compute()
        self.log("test_acc_step", acc, prog_bar=True)
        self.log("test_recall_step", rec, prog_bar=True)
        self.log("test_AUROC_step", auroc, prog_bar=True)
        self.log("test_precision_step", prec, prog_bar=True)
        self.log("test_f1_step", f1, prog_bar=True)

    def configure_optimizers(self):
        lr = self.args["lr"]
        b1 = self.args["b1"]
        b2 = self.args["b2"]

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(b1, b2))
        return [opt], []

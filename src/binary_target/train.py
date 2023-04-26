import sys,os
sys.path.append(os.getcwd())
import warnings
import pickle
import torch
import random
import logging
import pytorch_lightning as pl
from src.utils import conf_utils
from models.WindCNN import WindNet, WindNetPL
from datamodule import WindDataModule

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')


def train(conf_path):
    Configuration = conf_utils.Config()
    cfg = Configuration.load_json(conf_path)
    print("Reading dataset")

    trainer = pl.Trainer(max_epochs=cfg.hparams.max_epoch,
                         accelerator="gpu",
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         default_root_dir='logs',
                         )

    dm = WindDataModule(conf_path)

    Configuration = conf_utils.Config()
    cfg = Configuration.load_json(conf_path)
    net = WindNet(cfg)
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    model = WindNetPL(cfg, net=net, optimizer=optimizer, scheduler=scheduler)

    print("Training NN")    
    trainer.fit(model, dm)

   


if __name__ == "__main__":
    
    train(conf_path='./configs/train_configs/train_conf.json')


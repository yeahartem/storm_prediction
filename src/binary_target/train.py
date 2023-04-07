import sys
import os
import time
import warnings
import pickle
import json

import numpy as np
import torch
import random
import logging
import pytorch_lightning as pl
import utils

from contextlib import redirect_stdout
from models.WindCNN import WindNet, WindNetPL
from datamodule import get_stations, extract_splitted_data, WindDataModule

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')


def train(conf_path):

   
    max_epoch = 200
    args = {'lr': 1e-4, 'threshold': 0.5}

    print("Reading dataset")

    trainer = pl.Trainer(max_epochs=max_epoch,
                         accelerator="gpu",
                         benchmark=True,
                         check_val_every_n_epoch=1)

    dm = WindDataModule(conf_path)
    
    Configuration = utils.Config()
    cfg = Configuration.load_json(conf_path)
    net = WindNet(cfg)
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    model = WindNetPL(args, net=net, optimizer=optimizer, scheduler=scheduler)

    print("Initializing NN - done")
    print("Training NN")
    print("See, e.g., tensorboard")
    trainer.fit(model, dm)
    print("Training NN - done")
    last_log_folder = sorted(os.listdir('lightning_logs'), key=lambda x: int(x.split('_')[-1]))[-1]
    file = open(os.path.join('lightning_logs', last_log_folder,'transform.pkl'), 'wb')
    pickle.dump(dm.transform, file)
    file.close()
    with open(os.path.join('lightning_logs', last_log_folder, 'test_metrics.txt'), 'w') as f:
        with redirect_stdout(f):
            trainer.test(model, dm)


if __name__ == "__main__":
    train(conf_path='/wind/configs/train_conf.json')

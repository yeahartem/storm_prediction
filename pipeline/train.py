import sys
import os
import time
import warnings
import json
from typing import Any

import torch
import random
import logging
import pytorch_lightning as pl
from numpy import ndarray
from pandas import Series, DataFrame

sys.path.append(os.path.realpath('.'))
sys.path.append('../')
from contextlib import redirect_stdout
from src.models.WindCNN import WindNet, WindNetPL
from src.data_assemble.wrap_data import get_stations, extract_splitted_data, WindDataModule
from src.data_assemble.create_dataset import create_dataset, read_splits

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s-%(message)s')


def train(path="conf/train_conf.json"):
    with open(path) as fs:
        conf = json.load(fs)

    print(conf)
    if conf["if_generate_dataset"]:
        create_dataset(conf)

    nn_config_path = conf["nn_init_data"]["nn_config_path"]
    batch_size = conf["nn_init_data"]["batch_size"]
    max_epoch = conf["nn_training_data"]["max_epoch"]
    train_split_path = 'conf/splits/train.txt'  # TODO move to conf
    test_split_path = 'conf/splits/test.txt'
    path_to_save_train = path_to_save_train = conf["path_to_save"] + "train"

    print("Reading dataset")

    stations_list = get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                                 stations_allowed_path="conf/splits/time_split_stations.txt",
                                 max_lat=80.52, min_lat=36.38, max_lon=181.45, min_lon=32.12, max_height=300,
                                 min_height=-10)

    logging.info(f'Total stations: {len(stations_list)}')
    X_train, y_train = extract_splitted_data(conf["path_to_save"] + "train", stations_list)
    X_test, y_test = extract_splitted_data(conf["path_to_save"] + "test", stations_list)

    print(f"class balance train: {sum(y_train)/len(y_train)} test: {sum(y_test)/len(y_test)}")

    X = {"Train": X_train, "Val": X_test, "Test": X_test}
    y = {"Train": y_train, "Val": y_test, "Test": y_test}
    print("Reading - done")

    print("Initializing NN")
    with open(nn_config_path) as fs:
        args = json.load(fs)

    trainer = pl.Trainer(max_epochs=max_epoch,
                         accelerator="gpu",
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         precision=16,
                         )

    dm = WindDataModule(X=X, y=y, batch_size=batch_size, downsample=False)

    net = WindNet()
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    model = WindNetPL(args, net=net, optimizer=optimizer, scheduler=scheduler)

    print("Initializing NN - done")
    print("Training NN")
    print("See, e.g., tensorboard")
    trainer.fit(model, dm)
    print("Training NN - done")

    with open('data/test_metrics.txt', 'w') as f:
        with redirect_stdout(f):
            trainer.test(model, dm)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append('conf/train_conf.json')
    path_to_config = sys.argv[1]
    t1 = time.time()
    train(path=path_to_config)
    t2 = time.time()
    print("Total time: ", t2 - t1)

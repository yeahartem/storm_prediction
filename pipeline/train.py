import sys
import os
import time
import warnings
import json
import torch
import random
import logging
import pytorch_lightning as pl

sys.path.append(os.path.realpath('.'))
sys.path.append('../')
from contextlib import redirect_stdout
from src.models.WindCNN import WindNet, WindNetPL
from src.data_assemble.wrap_data import train_val_test_split, extract_splitted_data, WindDataModule
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
    st_split_dict = train_val_test_split(path_to_save_train, train=0.5, val=0.5, test=0.5, verbose=True)

    X_train, y_train = extract_splitted_data(conf["path_to_save"] + "train", st_split_dict)
    X_test, y_test = extract_splitted_data(conf["path_to_save"] + "test", st_split_dict)
    print("Reading - done")

    print("Initializing NN")
    with open(nn_config_path) as fs:
        args = json.load(fs)

    trainer = pl.Trainer(max_epochs=max_epoch,
                         # accelerator="cpu",
                         gpus=[0],
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         )

    dm = WindDataModule(X=X_train, y=y_train, batch_size=batch_size, downsample=False)
    model = WindNetPL(args)

    print("Initializing NN - done")

    print("Training NN")
    print("See, e.g., tensorboard")
    trainer.fit(model, dm)
    print("Training NN - done")

    dm = WindDataModule(X=X_test, y=y_test, batch_size=batch_size, downsample=False)
    print("Testing NN")
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

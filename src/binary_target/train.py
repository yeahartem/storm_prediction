import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import random
import logging
import pytorch_lightning as pl
from models.WindCNN import WindNet, WindNetPL
from datamodule import WindDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
os.environ['WANDB_DIR'] = 'outputs/wandb'
os.environ['WANDB_CONFIG_DIR'] = 'outputs/wandb'
os.environ['WANDB_CACHE_DIR'] = 'outputs/wandb'

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/train_configs"), config_name="train_conf")
def train(cfg: DictConfig) -> None:        

    logging.basicConfig(level=cfg.logging_level, format='%(asctime)s-%(message)s')

    wandb.init(project="Wind-speed",
               name=cfg.experiment_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               dir=os.path.join(os.getcwd(), "outputs/wandb"))

    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), "outputs/wandb"),
                               project="Wind-speed",
                               name=cfg.experiment_name)

    dm = WindDataModule(cfg)
    net = WindNet(cfg)
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    model = WindNetPL(cfg, net=net, optimizer=optimizer, scheduler=scheduler)
    wandb_logger.watch(model, log='all', log_freq=100)

    logging.info(f"Class balance train: {dm.class_balance_train}, class balance test: {dm.class_balance_test}")
    logging.info(f"Train size: {dm.train_size}, test size: {dm.test_size}")
    logging.info(f"Station count: {dm.station_count}")

    trainer = pl.Trainer(max_epochs=cfg.hparams.max_epoch,
                         accelerator="gpu",
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         default_root_dir=os.path.join(os.getcwd(), "outputs"),
                         logger=wandb_logger,
                         )       
    trainer.fit(model, dm)


if __name__ == "__main__":  
    train()


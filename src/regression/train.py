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
import time

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = 'outputs/wandb'
os.environ['WANDB_CONFIG_DIR'] = 'outputs/wandb'
os.environ['WANDB_CACHE_DIR'] = 'outputs/wandb'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/train_configs"), config_name="train_world_reg")
def train(cfg: DictConfig) -> None:        
    start_time = time.process_time()
    wandb.init(project=cfg.project_name,
               name=cfg.experiment_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               dir=os.path.join(os.getcwd(), "outputs/wandb"))

    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), "outputs/wandb"),
                               project=cfg.project_name,
                               name=cfg.experiment_name)

    dm = WindDataModule(cfg)
    net = WindNet(cfg)
    net = torch.compile(net)
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    criterion = torch.nn.MSELoss()

    model = WindNetPL(cfg, net=net, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
    wandb_logger.watch(model, log='all', log_freq=100)

    logging.info(f"Train size: {dm.train_size}, test size: {dm.test_size}")
    logging.info(f"Station count: {dm.station_count}")
    logging.info(f"Station count: {dm.station_count}")
    logging.info(f"Train rectangle: {dm.result_train_rectangle}")
    logging.info(f"Test rectangle: {dm.result_test_rectangle}")
    logging.info(f"Target min: {dm.min_target}, target max: {dm.max_target}")
    logging.info(f"Target mean: {dm.mean_target}, target std: {dm.std_target}")

    trainer = pl.Trainer(max_epochs=cfg.max_epoch,
                         accelerator="gpu",
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         default_root_dir=os.path.join(os.getcwd(), "outputs"),
                         logger=wandb_logger)       
    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)


if __name__ == "__main__":  
    train()


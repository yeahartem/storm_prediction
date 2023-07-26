import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import random
import logging
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from datamodule import WindDataModuleAlt
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, OnExceptionCheckpoint

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_DIR'] = 'out/wandb'
os.environ['WANDB_CONFIG_DIR'] = 'out/wandb'
os.environ['WANDB_CACHE_DIR'] = 'out/wandb'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
torch.set_float32_matmul_precision('high')


def test(cfg: DictConfig) -> None:        
    start_time = time.process_time()  
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), "outdb"),
                               project=cfg.project_name,
                               name=cfg.experiment_name)
    dm = WindDataModuleAlt(cfg)
    model = WindNetPL.load_from_checkpoint(os.path.join(os.getcwd(), "out", cfg.path_to_checkpoint), cfg=cfg)

    wandb_logger.watch(model, log='all', log_freq=100)       
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    default_root_dir = os.path.join(os.getcwd(), "out")#os.path.join(os.getcwd(), "out")
    trainer = pl.Trainer(max_epochs=cfg.max_epoch,
                         accelerator="gpu",
                         precision="16-mixed",
                         benchmark=True,
                         devices=[0],
                         check_val_every_n_epoch=1,
                         default_root_dir=default_root_dir,
                         logger=wandb_logger,
                         callbacks=[lr_monitor],) 

    logging.info(f"Time to start test {time.process_time() - start_time} seconds")
    trainer.test(model, dm)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/test_configs"), config_name="test_world_reg_test")
def main(cfg: DictConfig):    
    test(cfg)
    logging.info('Test finished!')


if __name__ == "__main__":      
    main()
    wandb.finish()
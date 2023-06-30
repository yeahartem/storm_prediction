import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import random
import logging
from datetime import datetime 
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from datamodule import WindDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, OnExceptionCheckpoint, ModelCheckpoint

warnings.filterwarnings("ignore")
torch.manual_seed(112)
random.seed(112)
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = 'outputs/wandb'
os.environ['WANDB_CONFIG_DIR'] = 'outputs/wandb'
os.environ['WANDB_CACHE_DIR'] = 'outputs/wandb'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
torch.set_float32_matmul_precision('high')


def train(cfg: DictConfig) -> None:        
    start_time = time.process_time()  
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), "outputs/wandb"),
                               project=cfg.project_name,
                               name=cfg.experiment_name)
    dm = WindDataModule(cfg)
    model = WindNetPL(cfg)

    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model)
    #     logging.info("Model compiled")
    # else:
    #     logging.info("PyTorch version is smaller than 2.0, compilation is not supported")
        
    wandb_logger.watch(model, log='all', log_freq=100)       
    default_root_dir = os.path.join(os.getcwd(), "outputs")
    checkpoint_loc = "outputs"    

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_loc, save_top_k=2, monitor="val/loss")
    # exception_checkpoint_callback = OnExceptionCheckpoint(checkpoint_loc)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False)
    
    trainer = pl.Trainer(max_epochs=cfg.max_epoch,                         
                         default_root_dir=default_root_dir,
                         callbacks=[lr_monitor, checkpoint_callback],
                         #performance
                         accelerator="gpu",
                         precision="16-mixed",
                         benchmark=True,
                         #validation
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0,
                         #distributed
                         devices=cfg.gpu_num,
                         num_nodes=cfg.num_nodes if cfg.distributed else 1,
                         strategy=cfg.strategy if cfg.distributed else 'auto',
                         #log
                         log_every_n_steps=cfg.log_every_n_steps,
                         logger=wandb_logger,
                         #misc
                         profiler='simple',
                         ) 
    
    # wandb.config.update(OmegaConf.to_container(cfg, resolve=True))    
    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)
    

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/train_configs"), config_name="train_world_reg_test")
def main(cfg: DictConfig):    
    train(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":      
    main()
    wandb.finish()

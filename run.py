import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import random
from tqdm import tqdm
import logging
import pytorch_lightning as pl
import pandas as pd
from src.regression.models.pl_module import WindNetPL
from src.regression.datamodule import WindDataInferModule
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, OnExceptionCheckpoint
from src.data_assemble.assemble_eval import load_dataset, load_target
from src.regression.data_load import DataPreLoader
from src.regression.models.pl_module import WindNetPL
import hydra
from omegaconf import DictConfig
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point, box
import geopandas as gpd
import fiona
from fiona.drvsupport import supported_drivers
from src.regression.eval import eval
from src.regression.risk_estimation import risk_estimation
supported_drivers['LIBKML'] = 'rw'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="cmip5_w_eval.yaml")
def main(cfg: DictConfig):    
    eval(cfg)
    risk_estimation(cfg)
    

if __name__ == "__main__":      
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()

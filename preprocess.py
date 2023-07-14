import sys,os
sys.path.append(os.getcwd())
import numpy as np
import xarray as xr
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.data_assemble.prepare_cmip5 import prepare_cmip

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="cmip5_w_eval.yaml")
def run_prepare(cfg: DictConfig):
    prepare_cmip(cfg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    run_prepare()
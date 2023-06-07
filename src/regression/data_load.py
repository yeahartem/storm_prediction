import sys,os
sys.path.append(os.getcwd())

import logging
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import xarray as xr
from src.data_assemble.assemble_data import make_blocks_no_target
from src.data_assemble.prepare_cmip5 import get_cmip5_files
from omegaconf import DictConfig
import dask
import dask.dataframe as dd
from dask.array.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars


def prepare_data(cfg):

    if not os.path.isfile('tmp_target.parquet'):

        logging.info('tmp file not found, processing')
        start_time = time.process_time()  
        files = get_cmip5_files(cfg.data_dir, cfg.variables)
        climate_file_paths = [file.path for file in files]
        logging.debug(f'loading {climate_file_paths}')   
        if cfg.data_in_ram:
            dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy').compute()
        else:
            dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy')
        # dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
        dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window, time_freq=cfg.time_freq)    
        logging.info(f"Time to load and prep climate data {time.process_time() - start_time} seconds")

        start_time = time.process_time()  

        target_df = polars.read_parquet(cfg.path_to_prepared_target_data).to_pandas()
        logging.debug(type(target_df['y'][0]))

        target_df['y_window'] = target_df['y'].rolling(cfg.time_window, center=True).max() 
        target_df = target_df.dropna()
        target_df = target_df.drop(columns=["y"])
        logging.info(f"Time to prepare target {time.process_time() - start_time} seconds")
        target_df.info()     

        # to indexes
        lat_intersection = np.intersect1d(dataset_as_blocks['lat'].data, target_df['lat'])
        lon_intersection = np.intersect1d(dataset_as_blocks['lon'].data, target_df['lon'])
        time_intersection = np.intersect1d(dataset_as_blocks['time'].data, target_df['time'])

        target_df = target_df.loc[target_df.lat.isin(lat_intersection)]
        target_df = target_df.loc[target_df.time.isin(time_intersection)]
        target_df = target_df.loc[target_df.lon.isin(lon_intersection)]
        target_df.to_parquet('tmp_target.parquet')
        logging.info('tmp file saved')

    else:
        logging.info('tmp file found')
        files = get_cmip5_files(cfg.data_dir, cfg.variables)
        climate_file_paths = [file.path for file in files]
        target_df = dd.read_parquet('tmp_target.parquet')
        dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy')
        # dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
        dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window, time_freq=cfg.time_freq)

    return dataset_as_blocks, target_df

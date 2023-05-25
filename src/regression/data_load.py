import sys,os
sys.path.append(os.getcwd())

import logging
import numpy as np
import xarray as xr
from src.data_assemble.assemble_data import make_blocks_no_target
from src.data_assemble.prepare_cmip5 import get_cmip5_files
import pandas as pd
import cudf
import hashlib

def hash_from_cfg(cfg):
    hashed_cfg =  int(hashlib.md5(frozenset(cfg.items())).hexdigest(), 16)
    return hash()


def prepare_data(cfg):
    
    files = get_cmip5_files(cfg.data_dir, cfg.variables)
    climate_file_paths = [file.path for file in files]
    logging.debug(f'loading {climate_file_paths}')

    if cfg.data_in_ram:
        dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy').compute()
    else:
        dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy')
        
    target_df = cudf.read_parquet(cfg.path_to_prepared_target_data)

    dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
    target_df['y_window'] = target_df['y'].rolling(window=cfg.time_window).max()
    target_df = target_df.drop(columns=["y", "height"]) 
    dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window, time_freq=cfg.time_freq)    

    # intersecting dataset and target_df    
    lat_intersection = np.intersect1d(dataset_as_blocks['lat'].data, target_df['lat'])
    target_df = target_df.loc[target_df.lat.isin(lat_intersection)]

    lon_intersection = np.intersect1d(dataset_as_blocks['lon'].data, target_df['lon'])
    target_df = target_df.loc[target_df.lon.isin(lon_intersection)]

    time_intersection = np.intersect1d(dataset_as_blocks['time'].data, target_df['time'])
    target_df = target_df.loc[target_df.time.isin(time_intersection)]

    target_df
    
    return dataset_as_blocks, target_df
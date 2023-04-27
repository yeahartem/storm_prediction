import sys,os
sys.path.append(os.getcwd())

import logging
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import pickle
import xarray as xr
from src.data_assemble.assemble_target import stations_to_data_grid
from src.data_assemble.assemble_data import make_blocks_no_target
from src.data_assemble.prepare_cmip5 import get_cmip5_files


import pandas as pd
from src.utils import conf_utils



def prepare_data(cfg):

    files = get_cmip5_files(cfg.data_dir, cfg.variables)
    climate_file_paths = [file.path for file in files]

    print(f'loading {climate_file_paths}')
    rectangle_coords = cfg['rectangle_coords']  

    dataset_xarray = xr.open_mfdataset(climate_file_paths,  combine="by_coords", parallel=True, engine='scipy')  

    target_df = pd.read_parquet(cfg.path_to_prepared_target_data)
    target_df['y'] = target_df['y'].astype('int')
    rectangle_coords = list(rectangle_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]

    stations_df = pd.read_parquet(cfg.path_to_prepared_stations)
    stations_df = stations_to_data_grid(dataset_xarray=dataset_xarray,
                                          stations_df=stations_df)
    target_df = target_df.merge(stations_df, on='station_name', how='left')
    target_df = target_df.drop(columns=["station_name", "height"])
     
    dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
    target_df['time'] = target_df['time'].astype('datetime64[D]')    

    target_df['y_window'] = target_df['y'].rolling(window=cfg.time_window).max()
    target_df = target_df.drop(columns=["y"]) 
    dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window)    

    # intersecting dataset and target_df
    lat_intersection = np.intersect1d(dataset_as_blocks['lat'].data, target_df['lat'])
    dataset_as_blocks = dataset_as_blocks.sel(lat=lat_intersection)
    target_df = target_df.loc[target_df.lat.isin(lat_intersection)]

    lon_intersection = np.intersect1d(dataset_as_blocks['lon'].data, target_df['lon'])
    dataset_as_blocks = dataset_as_blocks.sel(lon=lon_intersection)
    target_df = target_df.loc[target_df.lon.isin(lon_intersection)]

    time_intersection = np.intersect1d(dataset_as_blocks['time'].data, target_df['time'])
    dataset_as_blocks = dataset_as_blocks.sel(time=time_intersection)
    target_df = target_df.loc[target_df.time.isin(time_intersection)]

    return dataset_as_blocks, target_df


def split_train_test(X, y, start_of_test):

    split_date = pd.to_datetime(start_of_test)
    X_train = X.sel(time=slice(None, split_date))
    X_test = X.sel(time=slice(split_date, None))
    y_train = y.loc[y['time'] < split_date]
    y_test = y.loc[y['time'] >= split_date]

    return X_train, X_test, y_train, y_test



class XarrayDatasetBinary(Dataset):

    def __init__(self, dataset_as_blocks, target_df, dtype=torch.float32):   
        self.time_idxs = dataset_as_blocks['time'].data.searchsorted(target_df['time'].values )
        self.lat_idxs = dataset_as_blocks['lat'].data.searchsorted(target_df['lat'].values )
        self.lon_idxs = dataset_as_blocks['lon'].data.searchsorted(target_df['lon'].values )
        self.indexes = np.array((self.lat_idxs, self.lon_idxs , self.time_idxs))
        self.dataset_as_blocks = dataset_as_blocks
        self.target_df = target_df 
        self.dtype = dtype

    def __len__(self):
        return len(self.time_idxs)-1

    def __getitem__(self, idx):

        target_index = idx
        target = self.target_df['y_window'].iloc[target_index]
        y = torch.tensor(target, dtype=torch.int)
        x_index = self.indexes[:, idx]
        X = self.dataset_as_blocks[x_index[0],x_index[1],x_index[2]].data
        X = torch.tensor(X, dtype=self.dtype)

        return X, y
    

class WindDataModule(pl.LightningDataModule):

    def __init__(
            self, conf_path, downsample: bool = False
    ):
        super().__init__()
        Configuration = conf_utils.Config()
        self.cfg = Configuration.load_json(conf_path)
        
        dataset_as_blocks, target_df = prepare_data(self.cfg)

        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(dataset_as_blocks, target_df, self.cfg.start_of_test)
        
        
        # mean_channels = self.X_train.mean(dim=[0, 1, -1, -2])
        # std_channels = self.X_train.std(dim=[0, 1, -1, -2])

        # self.transform = transforms.Compose(
        #     [
        #         transforms.Normalize(mean=mean_channels, std=std_channels),
        #     ]
        # )


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = XarrayDatasetBinary(self.X_train, self.y_train)
            self.dataset_val = XarrayDatasetBinary(self.X_test, self.y_test)

        if stage == "test" or stage is None:
            self.dataset_test = XarrayDatasetBinary(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.hparams.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.hparams.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.hparams.batch_size, num_workers=12)
    

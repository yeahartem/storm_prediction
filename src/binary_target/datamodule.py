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
from omegaconf import DictConfig

import pandas as pd
from src.utils import conf_utils



def prepare_data(cfg):

    files = get_cmip5_files(cfg.data_dir, cfg.variables)
    climate_file_paths = [file.path for file in files]

    logging.debug(f'loading {climate_file_paths}')
    
    dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy')  

    target_df = pd.read_parquet(cfg.path_to_prepared_target_data)
    target_df["y"] = (target_df['y'] > cfg.speed_th)     
    target_df['y'] = target_df['y'].astype('int')

    dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
    target_df['y_window'] = target_df['y'].rolling(window=cfg.time_window).max()
    target_df = target_df.drop(columns=["y", "height"]) 
    dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window)    

    # intersecting dataset and target_df
    
    lat_intersection = np.intersect1d(dataset_as_blocks['lat'].data, target_df['lat'])
    target_df = target_df.loc[target_df.lat.isin(lat_intersection)]

    lon_intersection = np.intersect1d(dataset_as_blocks['lon'].data, target_df['lon'])
    target_df = target_df.loc[target_df.lon.isin(lon_intersection)]

    time_intersection = np.intersect1d(dataset_as_blocks['time'].data, target_df['time'])
    target_df = target_df.loc[target_df.time.isin(time_intersection)]

    return dataset_as_blocks, target_df


def split_train_test(X, y, start_of_test, test_coords):

    split_date = pd.to_datetime(start_of_test)
    X_train = X.sel(time=slice(None, split_date))
    X_test = X.sel(time=slice(split_date, None))
    y_train = y.loc[y['time'] < split_date]
    y_test = y.loc[y['time'] >= split_date]
    assert len(X_train.time) > 0
    assert len(X_test.time) > 0
    return X_train, X_test, y_train, y_test



class XarrayDatasetBinary(Dataset):

    def __init__(self, dataset_as_blocks, target_df, dtype=torch.float32, transforms_data=None):   

        self.target_df = target_df.dropna() 
        self.time_idxs = dataset_as_blocks['time'].data.searchsorted(self.target_df['time'].values )
        self.lat_idxs = dataset_as_blocks['lat'].data.searchsorted(self.target_df['lat'].values )
        self.lon_idxs = dataset_as_blocks['lon'].data.searchsorted(self.target_df['lon'].values )
        self.time_idxs = self.time_idxs[~np.isnan(self.time_idxs)] 
        self.lon_idxs = self.lon_idxs[~np.isnan(self.lon_idxs)] 
        self.lat_idxs = self.lat_idxs[~np.isnan(self.lat_idxs)] 
        assert len(self.time_idxs) == len(self.lat_idxs) == len(self.lon_idxs)
        assert len(self.time_idxs) == len(self.target_df), f'len(self.time_idxs)={len(self.time_idxs)}, len(self.target_df)={len(self.target_df)}'
        self.indexes = np.array((self.lat_idxs, self.lon_idxs , self.time_idxs))

        self.dataset_as_blocks = dataset_as_blocks        
        self.dtype = dtype
        self.transforms_data = transforms_data

    def __len__(self):
        return len(self.time_idxs)-1

    def __getitem__(self, idx):

        target_index = idx
        target = self.target_df['y_window'].iloc[target_index]
        y = torch.tensor(target, dtype=torch.int)
        x_index = self.indexes[:, idx]
        X = self.dataset_as_blocks[x_index[0],x_index[1],x_index[2]].data
        X = torch.tensor(X, dtype=self.dtype)
        if self.transforms_data:
            X = self.transforms_data(X)

        return X, y
    

class WindDataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        dataset_as_blocks, target_df = prepare_data(self.cfg)
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(dataset_as_blocks, target_df, self.cfg.start_of_test)        
        self.class_balance_train = self.y_train['y_window'].value_counts(normalize=True).values
        self.class_balance_test = self.y_test['y_window'].value_counts(normalize=True).values
        self.train_size = len(self.y_train)
        self.test_size = len(self.y_test)
        self.station_count = len(self.y_train['station_name'].unique())
        #get normalization values   
        mean_channels = np.load(self.cfg.path_to_means)
        std_channels = np.load(self.cfg.path_to_std)
        
        self.transform = transforms.Compose(
             [
                 transforms.Normalize(mean=mean_channels, std=std_channels),
             ]
        )


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = XarrayDatasetBinary(self.X_train, self.y_train, transforms_data=self.transform)
            self.dataset_val = XarrayDatasetBinary(self.X_test, self.y_test, transforms_data=self.transform)

        if stage == "test" or stage is None:
            self.dataset_test = XarrayDatasetBinary(self.X_test, self.y_test, transforms_data=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.batch_size, num_workers=4)
    

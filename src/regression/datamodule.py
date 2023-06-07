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


    logging.info('tmp file not found, processing')
    start_time = time.process_time()  
    files = get_cmip5_files(cfg.data_dir, cfg.variables)
    climate_file_paths = [file.path for file in files]
    logging.debug(f'loading {climate_file_paths}')   
    if cfg.data_in_ram:
        dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy').compute()
    else:
        dataset_xarray = xr.open_mfdataset(climate_file_paths, combine="by_coords", parallel=True, engine='scipy')
    dataset_xarray['time'] = dataset_xarray['time'].astype('datetime64[D]')
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

    return dataset_as_blocks, target_df



def split_train_test(X, y, start_of_test, test_coords):

    split_date = datetime.strptime(start_of_test, '%Y-%m-%d')
    X_train = X.sel(time=slice(None, split_date))
    X_test = X.sel(time=slice(split_date, None))
    logging.info(f'Target start {y.time.max()}')
    logging.info(f'Target start {y.time.max()}')
    y_train = y.loc[y['time'] < split_date]
    y_test = y.loc[y['time'] >= split_date]
    y_test = y.loc[y['lat'] >= test_coords.lat_min]
    y_test = y.loc[y['lat'] < test_coords.lat_max]
    y_test = y.loc[y['lat'] >= test_coords.lon_min]
    y_test = y.loc[y['lon'] < test_coords.lon_max]

    assert len(X_train.time) > 0
    assert len(X_test.time) > 0
    assert len(y_test) > 0

    return X_train, X_test, y_train, y_test



class XarrayDatasetBinary(Dataset):

    def __init__(self, dataset_as_blocks, target_df, dtype=torch.float32, transforms_data=None, target_max=1, target_min=0):   

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
        self.target_max = target_max
        self.target_min = target_min
        self.dataset_as_blocks = dataset_as_blocks        
        self.dtype = dtype
        self.transforms_data = transforms_data

    def __len__(self):
        return len(self.time_idxs)-1

    def __getitem__(self, idx):
        target_index = idx
        target = self.target_df['y_window'].iloc[target_index]
        y = torch.tensor(target, dtype=self.dtype)
        # y = torch.add(torch.div(y, self.target_max), -1.0 * self.target_min)  
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
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(dataset_as_blocks,
                                                                                target_df,
                                                                                self.cfg.start_of_test,
                                                                                self.cfg.test_coords)  
        self.train_size = len(self.y_train)
        self.test_size = len(self.y_test)
        # self.station_count = len(self.y_train['station_name'].unique())
        self.max_target = self.y_train['y_window'].max()
        self.min_target = self.y_train['y_window'].min()
        self.mean_target = self.y_train['y_window'].mean()
        self.std_target = self.y_train['y_window'].std()
 
        self.result_train_rectangle = (self.X_train.lat.min().data, self.X_train.lat.max().data, self.X_train.lon.min().data, self.X_train.lon.max().data)
        self.result_test_rectangle = (self.X_test.lat.min().data, self.X_test.lat.max().data, self.X_test.lon.min().data, self.X_test.lon.max().data)

        self.extreme_stations_train = (self.y_train.lat.min(), self.y_train.lat.max(), self.y_train.lon.min(), self.y_train.lon.max())
        self.extreme_stations_test= (self.y_test.lat.min(), self.y_test.lat.max(), self.y_test.lon.min(), self.y_test.lon.max())

        mean_channels = np.load(self.cfg.path_to_means)
        std_channels = np.load(self.cfg.path_to_std)
        
        self.transform = transforms.Compose(
             [
                 transforms.Normalize(mean=mean_channels, std=std_channels),
             ]
        )
        

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = XarrayDatasetBinary(self.X_train, self.y_train,
                                                     transforms_data=self.transform,
                                                     target_max=self.cfg.target_max,
                                                     target_min=self.cfg.target_min)
            self.dataset_val = XarrayDatasetBinary(self.X_test, self.y_test,
                                                   transforms_data=self.transform,
                                                   target_max=self.cfg.target_max,
                                                   target_min=self.cfg.target_min)

        if stage == "test" or stage is None:
            self.dataset_test = XarrayDatasetBinary(self.X_test, self.y_test,
                                                    transforms_data=self.transform,
                                                    target_max=self.cfg.target_max,
                                                    target_min=self.cfg.target_min)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    

if __name__ == '__main__':
    pass
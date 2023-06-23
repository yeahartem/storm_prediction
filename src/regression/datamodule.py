import sys,os
sys.path.append(os.getcwd())

import logging
import numpy as np
import pytorch_lightning as pl
import torchvision  
from torch.utils.data import DataLoader, Dataset
import torch
from src.data_assemble.assemble_data import make_blocks_numpy
from src.data_assemble.prepare_cmip5 import get_cmip5_files
from omegaconf import DictConfig
import dask
import dask.dataframe as dd
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars


def prepare_data(cfg):

    logging.info('tmp file not found, processing')
    start_time = time.process_time()
    time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
    lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
    lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
    var_data = np.empty((len(cfg.variables), len(time_coords), len(lat_coords), len(lon_coords)), dtype=np.float32)

    for i, var in enumerate(cfg.variables):
        var_data[i] = np.load(os.path.join(cfg.data_dir, var + '_' + str(cfg.precision) + '.npy'))

    var_data = np.moveaxis(var_data, 0, 1)
    dataset_as_blocks = make_blocks_numpy(var_data, cfg.half_side_size, time_stack_size=cfg.time_window, time_freq=cfg.time_freq)  

    logging.info(f"Time to load and prep climate data {time.process_time() - start_time} seconds")

    def max_window(values):
        if len(values)<cfg.time_window:
            return None
        else:
            return np.max(sliding_window_view(np.array(values), window_shape = cfg.time_window), axis = 1)
        
    def align_time(values):
        if len(values)<cfg.time_window:
            return None
        else:
            values = time_coords.searchsorted(values) # time into inds
            return np.array(values)[cfg.time_window//2:len(values) - cfg.time_window//2]
    
    def align_coords(values):
        lat, lon = values - cfg.half_side_size
        if (lat < 0) or (lon < 0):
            return None
        elif (lat > (len(lat_coords) - 2*cfg.half_side_size)) or (lon > (len(lon_coords) - 2*cfg.half_side_size)):
            return None
        else:
            return np.array((lat, lon))
        
    start_time = time.process_time()  
    target_df = polars.read_parquet(cfg.path_to_prepared_target_data)
    logging.info(f"Records before preparation {len(target_df)}")

    target_df = (
        target_df
        .lazy()        
        .sort("time")
        .groupby(["station_name"])
        .agg(
            [polars.col('time').apply(align_time), polars.col('y').apply(max_window)]
        )
        .collect()
    )

    target_df = target_df.with_columns(polars.col('station_name').apply(align_coords).keep_name())
    logging.info(f"Stations before droppping: {len(target_df)}")
    logging.info(f"Time to prepare target {time.process_time() - start_time} seconds")
    split_date = datetime.strptime(cfg.start_of_test, '%Y-%m-%d').date()
    split_index = time_coords.searchsorted(split_date)

    train_data_idxs = []
    test_data_idxs = []

    for coords, dates, y in target_df.rows():
        if (coords is not None) and (dates is not None) and (y is not None):
            if not (any(np.isnan(np.array(coords), casting='unsafe')) and any(np.isnan(np.array(dates), casting='unsafe')) and any(np.isnan(np.array(y), casting='unsafe'))):
                clipped_dates = dates[dates < (dataset_as_blocks.shape[2]-1)]
                dates_train = clipped_dates[clipped_dates < split_index]
                dates_test = clipped_dates[clipped_dates >= split_index]

                y_train = y[:len(dates_train)]
                y_test = y[len(dates_train):len(clipped_dates)]

                arr_train = np.stack([np.full(len(dates_train),coords[0], dtype=np.int16), np.full(len(dates_train), coords[1], dtype=np.int16), dates_train, y_train])
                arr_test = np.stack([np.full(len(dates_test),coords[0], dtype=np.int16), np.full(len(dates_test), coords[1], dtype=np.int16), dates_test, y_test])

                train_data_idxs.append(arr_train)
                test_data_idxs.append(arr_test)
    
    train_data_idxs = np.concatenate(train_data_idxs, axis=1)
    test_data_idxs = np.concatenate(test_data_idxs, axis=1)
    logging.info(f'Records prepared train {train_data_idxs.shape[1]}')
    logging.info(f'Records prepared test {test_data_idxs.shape[1]}')

    return dataset_as_blocks, train_data_idxs, test_data_idxs


class XarrayDataset(Dataset):

    def __init__(self, data_idxs, dataset_as_blocks, dtype=torch.float32, transforms=None):
        self.dataset_as_blocks = dataset_as_blocks
        self.data_idxs = data_idxs
        self.dtype = dtype
        self.transforms = transforms

    def __len__(self):
        return self.data_idxs.shape[1]

    def __getitem__(self, idx):
        lat, lon, date, y = self.data_idxs[:, idx]
        X = self.dataset_as_blocks[lat, lon, date]
        y = torch.tensor(y, dtype=self.dtype)
        # y = torch.add(torch.div(y, self.target_max), -1.0 * self.target_min)  
        X = torch.tensor(X, dtype=self.dtype)
        if self.transforms:
            X = self.transforms(X)
        return X, y
    

class WindDataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg        
        self.dataset_as_blocks, self.train_data_idxs, self.test_data_idxs = prepare_data(self.cfg)
        self.train_size = len(self.train_data_idxs)
        self.test_size = len(self.test_data_idxs)
        # self.station_count = len(self.y_train['station_name'].unique())
        # self.max_target = self.y_train['y_window'].max()
        # self.min_target = self.y_train['y_window'].min()
        # self.mean_target = self.y_train['y_window'].mean()
        # self.std_target = self.y_train['y_window'].std()
        # self.result_train_rectangle = (self.X_train.lat.min().data, self.X_train.lat.max().data, self.X_train.lon.min().data, self.X_train.lon.max().data)
        # self.result_test_rectangle = (self.X_test.lat.min().data, self.X_test.lat.max().data, self.X_test.lon.min().data, self.X_test.lon.max().data)
        # self.extreme_stations_train = (self.y_train.lat.min(), self.y_train.lat.max(), self.y_train.lon.min(), self.y_train.lon.max())
        # self.extreme_stations_test= (self.y_test.lat.min(), self.y_test.lat.max(), self.y_test.lon.min(), self.y_test.lon.max())
        
        if self.cfg.normalize:
            mean_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            std_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        else: self.transform = None
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = XarrayDataset(self.train_data_idxs, self.dataset_as_blocks,
                                                     transforms=self.transform)
            self.dataset_val = XarrayDataset(self.test_data_idxs, self.dataset_as_blocks,
                                                   transforms=self.transform)
        if stage == "test" or stage is None:
            self.dataset_test = XarrayDataset(self.test_data_idxs, self.dataset_as_blocks,
                                                    transforms=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    

if __name__ == '__main__':
    pass
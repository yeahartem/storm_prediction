import logging
import os
import sys
import numpy as np
from functools import partial
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch
import pickle
import xarray as xr
from src.data_assemble.assemble_target import stations_to_data_grid
from src.data_assemble.assemble_data import make_blocks_no_target

sys.path.append(os.path.join('/', 'wind', 'src', 'binary_target'))

import pandas as pd
from src.utils import conf_utils


# from sklearn.model_selection import train_test_split

def split_train_test(X, y, start_of_test):

    split_date = pd.to_datetime(start_of_test)
    X = pd.DataFrame.from_dict(X)
    y = pd.DataFrame.from_dict(y)
    X_train = X.loc[X['time'] < split_date]
    X_test = X.loc[X['time'] >= split_date]
    y_train = y.loc[y['time'] < split_date]
    y_test = y.loc[y['time'] >= split_date]

    return dict(X_train), dict(X_test), dict(y_train), dict(y_test)


def prepare_data(cfg):

    climate_file_paths = [os.path.join(cfg.data_dir, var + '.nc') for var in cfg.variables]
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
    target_df.set_index("time", inplace=True)

    time_intersection = np.intersect1d(dataset_xarray['time'].data, target_df.index)
    dataset_xarray = dataset_xarray.sel(time=time_intersection)
    target_df = target_df.loc[time_intersection]

    dataset_as_blocks = make_blocks_no_target(dataset_xarray, cfg.half_side_size, time_stack_size=cfg.time_window)
    target_df['y_window'] = target_df['y'].rolling(window=cfg.time_window).max()
    target_df = target_df.drop(columns=["y"])
    target_df.index = pd.to_datetime( target_df.index )

    target_df = target_df.pivot_table(index=["time", 'lat', 'lon'], values="y_window")
    ind = np.array(target_df.index, dtype=np.dtype('datetime64', 'float', 'float'))
    dataset_as_blocks = dataset_as_blocks.sel(target_df.index)

    1 == 1



class WindDataModule(pl.LightningDataModule):

    def __init__(
            self, conf_path, downsample: bool = False
    ):
        super().__init__()
        Configuration = conf_utils.Config()
        cfg = Configuration.load_json(conf_path)
        
        X, y = prepare_data(cfg)

        X_train, X_test, y_train, y_test = split_train_test(X, y, cfg.start_of_test)
        
        self.X_train, self.X_val, self.X_test = (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.tensor(y_train, dtype=torch.int),
            torch.tensor(y_test, dtype=torch.int),
            torch.tensor(y_test, dtype=torch.int),
        )

        self.batch_size = cfg.nn_training_data.batch_size

        mean_channels = self.X_train.mean(dim=[0, 1, -1, -2])
        std_channels = self.X_train.std(dim=[0, 1, -1, -2])

        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=mean_channels, std=std_channels),
            ]
        )
        self.dl_dict = {"batch_size": self.batch_size}

    
    def prepare_data(self):
        if type(self.y_train) == torch.Tensor and len(self.y_train.shape) == 2:
            pass
        else:
            self.y_train = torch.tensor(self.y_train)
            self.y_val = torch.tensor(self.y_val)
            self.y_test = torch.tensor(self.y_test)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(
                self.transform(self.X_train), torch.tensor(self.y_train)
            )
            self.dataset_val = TensorDataset(
                self.transform(self.X_val), torch.tensor(self.y_val)
            )

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(
                self.transform(self.X_test), torch.tensor(self.y_test)
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, sampler=self.sampler, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, sampler=self.sampler, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, sampler=self.sampler, **self.dl_dict)
    

def extract_splitted_data(path_to_dump: str, sts: list) -> tuple:
    """extract data from listed stations
    Args:
        path_to_dump (str): path to folder which contains folder with preparsed numpy objects from stations
        sts (list): list of stations to be extracted
    Returns:
        X (np.array) : data
        y (np.array) : targets
    """
    X = []
    y = []
    for st in sts:
        logging.debug(f'extracting {st}')
        st_dir = os.path.join(path_to_dump, st)
        try:
            with open(os.path.join(st_dir, "objects.npy"), "rb") as f:
                X_ = np.load(f, allow_pickle=True).astype(np.float32)
                X.append(X_)
        except FileNotFoundError:
            logging.warning(f'{st} not found')
            continue
        try:
            with open(os.path.join(st_dir, "target.npy"), "rb") as f:
                y_ = np.load(f, allow_pickle=True)
            y.append(y_)
        except FileNotFoundError:
            # logging.warning(f'{st} empty target')
            continue

    if X:
        X = np.concatenate(X)
        y = np.concatenate(y)
    else:
        logging.critical('Train data is empty')
    return X, y

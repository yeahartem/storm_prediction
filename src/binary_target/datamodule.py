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
from tqdm import tqdm
import xarray as xr

sys.path.append(os.path.join('/', 'wind', 'src', 'binary_target'))

import geopandas as gpd
from shapely.geometry import Point, Polygon, box, LineString

import pandas as pd
import matplotlib.pyplot as plt
from assemble_conv import load_weatherstation_df, get_y, get_pixel_stations, make_blocks_numpy, assemble_numpy_ds
import utils
# from sklearn.model_selection import train_test_split

def split_train_test(X, y, cfg):
    split_time = cfg.start_of_test
    X_train_d = {}
    X_test_d = {}
    y_train_d = {}
    y_test_d = {}
    for k in X.keys():
        y_train_d[k] = y[k].loc[slice(None, split_time)].dropna() #strange nans in the middle appear
        t_axis_train = y_train_d[k].index
        t_axis_train = np.intersect1d(t_axis_train, X[k].time.data)
        y_test_d[k] = y[k].loc[slice(split_time, None)].dropna()
        t_axis_test = y_test_d[k].index
        t_axis_test = np.intersect1d(t_axis_test, X[k].time.data)
        

        X_train_d[k] = X[k].sel(time=t_axis_train).data
        X_test_d[k]  = X[k].sel(time=t_axis_test).data

        
        assert (np.isnan(y_train_d[k]).sum().values[0] == 0) and ((np.isnan(y_test_d[k]).sum().values[0] == 0))
        assert (X_train_d[k].shape[0] == y_train_d[k].shape[0]) and (X_test_d[k].shape[0] == y_test_d[k].shape[0])

    X_train = np.concatenate(list(X_train_d.values()))
    X_test = np.concatenate(list(X_test_d.values()))

    y_train = np.concatenate(list(y_train_d.values()))
    y_test = np.concatenate(list(y_test_d.values()))

    return X_train, X_test, y_train, y_test

def prepare_data(cfg):

    climate_file_paths = [os.path.join(cfg.data_dir, var + '.nc') for var in cfg.variables]
    print(f'loading {climate_file_paths}')
    lat_lon_bnds = cfg['rectangle_coords']
    time_bnds = cfg['time_limits']
    def _cut_lan_lot_time(x, lat_lon_bnds, time_bnds):
        return x.sel(lon=slice(*(lat_lon_bnds['lon_min'], lat_lon_bnds['lon_max'])), lat=slice(*(lat_lon_bnds['lat_min'], lat_lon_bnds['lat_max'])), time=slice(*(time_bnds[0], time_bnds[1])))
    _cut = partial(_cut_lan_lot_time, lat_lon_bnds=lat_lon_bnds, time_bnds=time_bnds)
    dataset_as_xarray = xr.open_mfdataset(climate_file_paths,  combine="by_coords", parallel=True, engine='scipy', compat='override', preprocess=_cut)
    var_names_to_drop = [v for v in dataset_as_xarray.keys() if v not in cfg.variables + ['lat', 'lon', 'time'] ] 
    coords_names_to_drop = [v for v in dataset_as_xarray.coords.keys() if v not in cfg.variables + ['lat', 'lon', 'time'] ] 
    dataset_as_xarray = dataset_as_xarray.drop_vars(var_names_to_drop)
    dataset_as_xarray = dataset_as_xarray.drop_vars(coords_names_to_drop)
    dataset_as_blocks = make_blocks_numpy(dataset_as_xarray, cfg.half_side_size)

    df = load_weatherstation_df(cfg.path_to_weather_stations_data)
    stations_list = get_stations(all_stations_data=cfg.path_to_weather_station_list,
                                    stations_allowed_path=cfg.path_to_allowed_stations,
                                    max_lat=dataset_as_blocks.lat.max().data,
                                    min_lat=dataset_as_blocks.lat.min().data,
                                    max_lon=dataset_as_blocks.lon.max().data,
                                    min_lon=dataset_as_blocks.lon.min().data,
                                    max_height=cfg.max_height,
                                    min_height=cfg.min_height)
    if not stations_list:
        raise ValueError('No stations found in the given area')

    weatherstation_list = pd.read_json(cfg.path_to_weather_station_list)
    weatherstation_list['Наименование станции'] = weatherstation_list['Наименование станции'].str.casefold()

    target = get_y(weather_stations_data=df,
                    start=cfg.time_limits[0],
                    end=cfg.time_limits[1],
                    station_name_list=stations_list,
                    speed_th=cfg.speed_th)


    stations_pixs = get_pixel_stations(dataset=dataset_as_blocks,
                                                station_names=stations_list,
                                                station_list=weatherstation_list)

    X, y = assemble_numpy_ds(dataset_as_blocks, target, stations_pixs)

    return X, y

class WindDataModule(pl.LightningDataModule):
    def __init__(
            self, conf_path, downsample: bool = False
    ):
        super().__init__()
        Configuration = utils.Config()
        cfg = Configuration.load_json(conf_path)
        if ("presaved_X_path" in cfg.keys()) and ("presaved_y_path" in cfg.keys()):
            file = open(cfg.presaved_X_path, 'rb')
            X = pickle.load(file)
            file.close()

            file = open(cfg.presaved_y_path, 'rb')
            y = pickle.load(file)
            file.close()
        else:
            X, y = prepare_data(cfg)
            file = open('X.pkl', 'wb')
            pickle.dump(X, file)
            file.close()
            file = open('y.pkl', 'wb')
            pickle.dump(y, file)
            file.close()



        X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)
        
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

        if downsample:
            def make_weights_for_balanced_classes(images, nclasses):
                n_images = len(images)
                count_per_class = [0] * nclasses
                for _, image_class in images:
                    count_per_class[image_class] += 1
                weight_per_class = [0.] * nclasses
                for i in range(nclasses):
                    weight_per_class[i] = float(n_images) / float(count_per_class[i])
                weights = [0] * n_images
                for idx, (image, image_class) in enumerate(images):
                    weights[idx] = weight_per_class[image_class]
                return weights

            weights = make_weights_for_balanced_classes(self.X_train, 2)
            weights = torch.tensor(weights, dtype=self.X_train.dtype, device=self.X_train.device)
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        else:
            self.sampler = None

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


def get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                 stations_allowed_path="conf/splits/time_split_stations.txt",
                 max_lat=None, min_lat=None, max_lon=None, min_lon=None, max_height=None, min_height=None):
    all_stations = pd.read_json(all_stations_data)
    with open(stations_allowed_path) as f:
        stations_allowed = f.read().split('\n')
    result_stations = all_stations[all_stations['Наименование станции'].isin(stations_allowed)]
    if max_lat:
        result_stations = result_stations[result_stations['Широта'] < max_lat]
    if min_lat:
        result_stations = result_stations[result_stations['Широта'] > min_lat]
    if max_lon:
        result_stations = result_stations[result_stations['Долгота'] < max_lon]
    if min_lon:
        result_stations = result_stations[result_stations['Долгота'] > min_lon]
    if max_height:
        result_stations = result_stations[result_stations['Высота метеопл.'] < max_height]
    if min_height:
        result_stations = result_stations[result_stations['Высота метеопл.'] > min_height]
    return list(result_stations['Наименование станции'].str.casefold())


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

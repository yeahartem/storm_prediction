import gc
import pickle
import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import random

sys.path.append(os.path.join('/', 'wind'))

from src.binary_target.assemble_conv import get_y, get_pixel_stations, make_blocks_numpy, assemble_numpy_ds
from src.binary_target.datamodule import get_stations
from src.data_assemble.data_processing import get_xarrays, get_xarrays_elevation
from src.binary_target.utils import cleanup_ms_name
from src.binary_target.utils import Config

warnings.filterwarnings("ignore")

torch.manual_seed(112)
random.seed(112)


def load_dataset_as_xarray(cmip_file_paths, elevation_path, rectangle_coords, target_res, bands, time_limits):
    cmip_data = get_xarrays(
        all_cmip_files=cmip_file_paths,
        rectangle_coords=rectangle_coords,
        bands=bands,
        target_res=target_res,
        time_limits=time_limits)
    cmip_xarray = cmip_data['Wind_']
    elevation = get_xarrays_elevation(
        path_to_data=elevation_path,
        rectangle_coords=rectangle_coords,
        reference_xarray=cmip_xarray)
    cmip_data.update(elevation)
    return cmip_data


def load_weatherstation_df(path_to_weather_stations):
    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"]
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
    logging.info("data_meteo_full is read")
    return df


def create_dataset(conf):
    time_limits = [np.datetime64(conf.time_limits[0]), np.datetime64(conf.time_limits[1])]
    split_date = pd.to_datetime(conf.start_of_test)

    df = load_weatherstation_df(conf.path_to_weather_stations_data)
    all_cmip_files = [os.path.join(conf.path_to_files[1], fn) for fn in next(os.walk(conf.path_to_files[1]))[2]]

    block_size = 10
    for lat in range(int(conf.rectangle_coords.lat_min), int(conf.rectangle_coords.lat_max), block_size):
        for lon in range(int(conf.rectangle_coords.lon_min), int(conf.rectangle_coords.lon_max), block_size):

            lat_max = min(lat + block_size, conf.rectangle_coords.lat_max)
            lon_max = min(lon + block_size, conf.rectangle_coords.lon_max)

            stations_list = get_stations(all_stations_data=conf.path_to_weather_station_list,
                                         stations_allowed_path=conf.path_to_allowed_stations,
                                         max_lat=lat_max, min_lat=lat, max_lon=lon_max, min_lon=lon,
                                         max_height=conf.max_height, min_height=conf.min_height)
            if not stations_list:
                logging.info('No station in this area')
                continue
            logging.info(f'stations to be created: {stations_list}')
            logging.info(f'working in rectangle: {lat, lon}')
            weatherstation_list = pd.read_json(conf.path_to_weather_station_list)
            target = get_y(weather_stations_data=df,
                           start=conf.time_limits[0],
                           end=conf.time_limits[1],
                           station_name_list=stations_list,
                           speed_th=conf.speed_th)
            dataset_as_xarray = load_dataset_as_xarray(cmip_file_paths=all_cmip_files,
                                                       elevation_path=conf.path_to_files[0],
                                                       rectangle_coords=[lat, lat_max, lon, lon_max],
                                                       target_res=conf.target_res,
                                                       bands=conf.bands,
                                                       time_limits=time_limits)
            stations_pixs = get_pixel_stations(dataset=dataset_as_xarray,
                                               station_names=stations_list,
                                               station_list=weatherstation_list)

            blocks = make_blocks_numpy(dataset_as_xarray=dataset_as_xarray,
                                       half_side_size=conf.half_side_size)
            X, y = assemble_numpy_ds(blocks, target, stations_pixs)
            logging.debug(f"Assembling dataset for training - done")

            path_to_dump = conf.path_to_save
            for k in X.keys():
                X_station_train = X[k][X[k].time < split_date]
                y_station_train = y[k][y[k].index < split_date]

                # Split by date on two groups (train and test), make path_to_save_train and nn_test
                st_path_train = os.path.join(conf.path_to_save, "train", k)
                if not os.path.isdir(st_path_train):
                    os.makedirs(st_path_train)

                with open(os.path.join(st_path_train, 'objects.npy'), 'wb') as f:
                    pickle.dump(X_station_train, f)
                with open(os.path.join(st_path_train, 'target.npy'), 'wb') as f:
                    pickle.dump(y_station_train, f)

                X_station_test = X[k][X[k].time >= split_date]
                y_station_test = y[k][y[k].index >= split_date]

                st_path_test = os.path.join(conf.path_to_save, 'test', k)
                if not os.path.isdir(st_path_test):
                    os.makedirs(st_path_test)

                with open(os.path.join(st_path_test, 'objects.npy'), 'wb') as f:
                    pickle.dump(X_station_test, f)
                with open(os.path.join(st_path_test, 'target.npy'), 'wb') as f:
                    pickle.dump(y_station_test, f)
                logging.debug(f'{k} station saved')
                gc.collect()


if __name__ == "__main__":
    logging.basicConfig(filename='dataset.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    Configuration = Config()
    config = Configuration.load_json('configs/train_conf.json')
    create_dataset(config)

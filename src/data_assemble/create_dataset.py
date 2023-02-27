import gc
import json
import pickle
import time
import os
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import random

from src.data_assemble.assemble_conv import get_y, get_pixel_stations, make_blocks, assemble_numpy_ds
from src.data_assemble.wrap_data import get_stations
from src.data_utils.data_processing import get_xarrays, get_xarrays_elevation
from src.data_utils.utils import cleanup_ms_name

warnings.filterwarnings("ignore")


torch.manual_seed(112)
random.seed(112)


def load_dataset_as_xarray(file_paths, rectangle_coords, target_res, bands, time_limits):
    cmip_data = get_xarrays(
        path_to_cmip_folder=file_paths[1],
        rectangle_coords=rectangle_coords,
        bands=bands,
        target_res=target_res,
        time_limits=time_limits)
    cmip_xarray = cmip_data['Wind_']
    elevation = get_xarrays_elevation(
        path_to_data=file_paths[0],
        rectangle_coords=rectangle_coords,
        reference_xarray=cmip_xarray)
    cmip_data.update(elevation)
    return cmip_data


def create_dataset(conf):
    path_to_files = conf["path_to_files"]
    half_side_size = conf["half_side_size"]
    rectangle_coords = conf["rectangle_coords"]
    max_lat = rectangle_coords['lat_max']
    min_lat = rectangle_coords['lat_min']
    max_lon = rectangle_coords['lon_max']
    min_lon = rectangle_coords['lon_min']
    min_height = conf["min_height"]
    max_height = conf["max_height"]
    target_res = conf["target_res"]
    time_limits = conf["time_limits"].deepcopy()
    time_limits[0] = np.datetime64(time_limits[0])
    time_limits[1] = np.datetime64(time_limits[1])
    start_of_test = conf["start_of_test"]
    split_date = pd.to_datetime(start_of_test['t_split'])
    path_to_weather_stations = conf["path_to_weather_stations"]
    path_to_weatherstation_list = conf["path_to_weather_stations_list"]
    path_to_save_train = conf["path_to_save"] + "train"
    path_to_save_test = conf["path_to_save"] + "test"
    speed_th = conf["nn_init_data"]["speed_th"]
    dataset_save_path = conf['path_to_save']
    bands = conf['bands']

    path_to_cmip = path_to_files[1]

    block_size = 10
    logging.info("Preparing target")
    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"]
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
    logging.info("data_meteo_full is read")

    for lat in range(int(min_lat), int(max_lat) + block_size, block_size):
        for lon in range(int(min_lon), int(max_lon) + block_size, block_size):
            lat = min(lat, max_lat)
            lon = min(lon, max_lon)
            rectangle_coords['lat_max'] = lat + block_size
            rectangle_coords['lat_min'] = lat
            rectangle_coords['lon_max'] = lon + block_size
            rectangle_coords['lon_min'] = lon
            stations_list = get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                                         stations_allowed_path="conf/splits/time_split_stations.txt",
                                         max_lat=lat + block_size, min_lat=lat, max_lon=lon + block_size, min_lon=lon,
                                         max_height=max_height, min_height=min_height)
            if not stations_list:
                logging.info('No station in this area')
                continue
            logging.info(f'stations to be created: {stations_list}')
            logging.info(f'working in rectangle: {rectangle_coords}')
            weatherstation_list = pd.read_json(path_to_weatherstation_list)
            target = get_y(weather_stations_data=df,
                           start=time_limits[0],
                           end=time_limits[1],
                           station_name_list=stations_list,
                           speed_th=speed_th)

            dataset_as_xarray = load_dataset_as_xarray(file_paths=path_to_files,
                                                 rectangle_coords=rectangle_coords,
                                                 target_res=target_res,
                                                 bands=bands,
                                                 time_limits=time_limits)

            stations_pixs = get_pixel_stations(dataset=dataset_as_xarray,
                                               station_names=stations_list,
                                               station_list=weatherstation_list)
            logging.debug(f"Preparing target - done")

            blocks = make_blocks(dataset_as_xarray=dataset_as_xarray,
                                 half_side_size=half_side_size)
            logging.debug(f"Preparing blocks - done")

            X, y = assemble_numpy_ds(blocks, target, stations_pixs)
            logging.debug(f"Assembling dataset for training - done")

            if dataset_save_path:
                path_to_dump = os.path.join('..', dataset_save_path)
            for k in X.keys():
                X_station_train = X[k][X[k].time < split_date]
                y_station_train = y[k][y[k].index < split_date]

                # Split by date on two groups (train and test), make path_to_save_train and nn_test
                st_path_train = os.path.join(path_to_save_train, k)
                if not os.path.isdir(st_path_train):
                    os.makedirs(st_path_train)

                with open(os.path.join(st_path_train, 'objects.npy'), 'wb') as f:
                    pickle.dump(X_station_train, f)
                with open(os.path.join(st_path_train, 'target.npy'), 'wb') as f:
                    pickle.dump(y_station_train, f)

                X_station_test = X[k][X[k].time >= split_date]
                y_station_test = y[k][y[k].index >= split_date]

                st_path_test = os.path.join(path_to_save_test, k)
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
    path_to_config = 'conf/train_conf.json'

    t1 = time.process_time()
    with open(path_to_config) as fs:
        config_dict = json.load(fs)

    create_dataset(config_dict)
    t2 = time.process_time()
    print("Total time: ", t2 - t1)

import gc
import json
import time
from itertools import islice
import numpy as np
import os
import sys
import logging
import warnings

sys.path.append('../')
sys.path.append(os.path.realpath('.'))
warnings.filterwarnings("ignore")
from src.data_assemble.assemble_conv import *
from src.data_assemble.wrap_data import *

torch.manual_seed(112)
random.seed(112)


def read_splits(train_path, val_path, test_path):
    with open(train_path) as f:
        train_list = f.read().split('\n')
    with open(test_path) as f:
        test_list = f.read().split('\n')
    if val_path:
        with open(test_path) as f:
            val_list = f.read().split('\n')
    else:
        val_list = test_list
    return train_list, val_list, test_list


def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def cleanup_ms_name(name: str):
    name = name.replace('"', '')
    name = name.replace(',', '')
    return name


def create_dataset(conf):
    path_to_files = conf["path_to_files"]
    half_side_size = conf["half_side_size"]
    rectangle_coords = conf["rectangle_coords"]
    min_height = conf["min_height"]
    max_height = conf["max_height"]
    target_res = conf["target_res"]
    filter_dict = conf["filter_dict"]
    time_limits = conf["time_limits"]
    start_of_test = conf["start_of_test"]
    start = time_limits['t_start']
    end = time_limits['t_end']
    split_date = pd.to_datetime(start_of_test['t_split'])
    path_to_weather_stations = conf["path_to_weather_stations"]
    path_to_weatherstation_list = conf["path_to_weather_stations_list"]
    path_to_save_train = conf["path_to_save"] + "train"
    path_to_save_test = conf["path_to_save"] + "test"
    speed_th = conf["nn_init_data"]["speed_th"]
    train_split_path = 'conf/splits/train.txt'  # TODO move to conf
    test_split_path = 'conf/splits/test.txt'
    dataset_save_path = conf['path_to_save']

    stations_list = get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                                 stations_allowed_path="conf/splits/time_split_stations.txt",
                                 max_lat=rectangle_coords['lat_max'],
                                 min_lat=rectangle_coords['lat_min'],
                                 max_lon=rectangle_coords['lon_max'],
                                 min_lon=rectangle_coords['lon_min'],
                                 max_height=300,
                                 min_height=-10)
    print('stations to be created: ', stations_list)
    for path in path_to_files:
        if 'elevation' not in path_to_files:
            path_to_cmip = path

    logging.info("Preparing target")
    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"]

    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    # logging.debug(df.dtypes)
    # logging.debug(df.memory_usage(deep=True))

    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    # df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    # df[["Максимальная скорость", "Средняя скорость ветра"]] =
    # df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
    logging.info("data_meteo_full is read")

    weatherstation_list = pd.read_json(path_to_weatherstation_list)
    i = 0
    target = get_y(df, start, end, stations_list, speed_th=speed_th)
    logging.debug(f'Target acquired')
    stations_pixs = get_pixel_stations(path_to_cmip, filter_dict, stations_list, weatherstation_list,
                                       rectangle_coords,
                                       target_res)
    logging.debug(f"Preparing target - done")

    logging.info(f"Preparing blocks")
    blocks = make_blocks(path_to_files, filter_dict, rectangle_coords, target_res, half_side_size=half_side_size,
                         time_limits=time_limits)
    logging.debug(f"Preparing blocks - done")

    logging.debug(f"Assembling dataset for training")
    X, y = assemble_numpy_ds(blocks, target, stations_pixs)
    logging.debug(f"Assembling dataset for training - done")

    with open('data_mounted/X_backup.npy', 'wb') as f:
        pickle.dump(X, f)
    with open('data_mounted/y_backup.npy', 'wb') as f:
        pickle.dump(y, f)

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


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s-%(message)s')

    if len(sys.argv) == 1:
        sys.argv.append('conf/train_conf.json')
    path_to_config = sys.argv[1]
    t1 = time.time()

    with open(path_to_config) as fs:
        conf = json.load(fs)

    create_dataset(conf)
    t2 = time.time()
    print("Total time: ", t2 - t1)

import gc
from itertools import islice
import numpy as np
import os
import sys
import logging
sys.path.append('../')
sys.path.append(os.path.realpath('.'))
import warnings

warnings.filterwarnings("ignore")
from src.data_assemble.assemble_conv import *
from src.data_assemble.wrap_data import *

torch.manual_seed(112)
random.seed(112)


def read_splits(train_path, test_path):
    with open(train_path) as f:
        train_list = f.read().split('\n')
    with open(test_path) as f:
        test_list = f.read().split('\n')
    return train_list, test_list


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

    station_names_train, station_names_test = read_splits(os.path.join(train_split_path),
                                                          os.path.join(test_split_path))
    station_names = station_names_train + station_names_test

    for path in path_to_files:
        if 'elevation' not in path_to_files:
            path_to_cmip = path

    logging.info("Preparing target")
    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"]

    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    # logging.debug(df.dtypes)
    # logging.debug(df.memory_usage(deep=True))
    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df[["Максимальная скорость", "Средняя скорость ветра"]] = df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
    logging.info("data_meteo_full is read")
    weatherstation_list = pd.read_json(path_to_weatherstation_list)
    # for i, stations_batch in enumerate(batched(station_names, 10)):
    i = 0
    stations_batch = station_names
    target = get_y(df, start, end, stations_batch, speed_th=speed_th)
    logging.debug(f'Target acquired, batch {i+1}')
    stations_pixs = get_pixel_stations(path_to_cmip, filter_dict, stations_batch, weatherstation_list,
                                       rectangle_coords,
                                       target_res)
    logging.debug("Preparing target - done, batch {i+1}")

    logging.info("Preparing blocks, batch {i+1}")
    blocks = make_blocks(path_to_files, filter_dict, rectangle_coords, target_res, half_side_size=half_side_size,
                         time_limits=time_limits)
    logging.debug("Preparing blocks - done, batch {i+1}")

    logging.debug("Assembling dataset for training, batch {i+1}")
    X, y = assemble_numpy_ds(blocks, target, stations_pixs)
    logging.debug("Assembling dataset for training - done, batch {i+1}")
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
            # np.save(f, X_station)
        with open(os.path.join(st_path_train, 'target.npy'), 'wb') as f:
            pickle.dump(y_station_train, f)
            # np.save(f, y_station)

        X_station_test = X[k][X[k].time >= split_date]
        y_station_test = y[k][y[k].index >= split_date]

        st_path_test = os.path.join(path_to_save_test, k)
        if not os.path.isdir(st_path_test):
            os.makedirs(st_path_test)

        with open(os.path.join(st_path_test, 'objects.npy'), 'wb') as f:
            pickle.dump(X_station_test, f)
            # np.save(f, X_station)
        with open(os.path.join(st_path_test, 'target.npy'), 'wb') as f:
            pickle.dump(y_station_test, f)
            # np.save(f, y_station)
        logging.debug("Batch {i+1} ended")

        gc.collect()

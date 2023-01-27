import numpy as np
import sys
sys.path.append('../')
import os
import torch
import time
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
import json
warnings.filterwarnings("ignore")
import xarray as xr
import geopandas as gpd
from contextlib import redirect_stdout

from src.data_assemble.assemble_ml import *
from src.data_assemble.assemble_conv import *
from src.models.utils import *
from src.data_utils.data_processing import *
from src.data_assemble.wrap_data import *
from src.models.WindCNN import *
from src.data_assemble.wrap_data import *
torch.manual_seed(112)
random.seed(112)


def infer(path_to_config="infer_conf.json"):
    with open(path_to_config) as jf:
        conf = json.load(jf)

    path_to_files = conf["path_to_files"]
    half_side_size = conf["half_side_size"]
    rectangle_coords = conf["rectangle_coords"]
    if_generate_dataset = conf["if_generate_dataset"]
    # presave_dataset = conf["presave_dataset"]
    target_res = conf["target_res"]
    filter_dict = conf["filter_dict"]
    time_limits = conf["time_limits"]
    start_of_test = conf["start_of_test"]
    start = time_limits['t_start']
    end = time_limits['t_end']
    split_date = pd.to_datetime(start_of_test['t_split'])
    path_to_weather_stations = conf["path_to_weather_stations"]
    path_to_weatherstation_list = conf["path_to_weather_stations_list"]
    # time_limits = {'t_start': np.datetime64(time_limits['t_start']), 't_end': np.datetime64(time_limits['t_end'])}
    station_names = conf["station_names"]
    nn_config_path = conf["nn_init_data"]["nn_config_path"]
    path_to_save_train = conf["path_to_save_train"]
    path_to_save_test = conf["path_to_save_test"]
    batch_size = conf["nn_init_data"]["batch_size"] 
    speed_th = conf["nn_init_data"]["speed_th"]
    max_epoch = conf["nn_training_data"]["max_epoch"]
    
    for path in path_to_files:
        if 'elevation' not in path_to_files:
            path_to_cmip = path
    
    if if_generate_dataset:
        print("Preparing target")
        df = pd.read_csv(path_to_weather_stations)
        print("data_meteo_full.csv is read")
        target = get_y(df, start, end, station_names, speed_th=speed_th)
        weatherstation_list = pd.read_csv(path_to_weatherstation_list)
        stations_pixs = get_pixel_stations(path_to_cmip, filter_dict, station_names, weatherstation_list, rectangle_coords, target_res)
        print("Preparing target - done")

        print("Preparing blocks")
        blocks = make_blocks(path_to_files, filter_dict, rectangle_coords, target_res, half_side_size=half_side_size, time_limits=time_limits)
        print("Preparing blocks - done")

        print("Assembling dataset for training")
        X, y = assemble_numpy_ds(blocks, target, stations_pixs)
        print("Assembling dataset for training - done")
        # if presave_dataset:
        # path_to_dump = os.path.join('..', 'data','nn_train')    # Redirect to STASH
        for k in X.keys():
            X_station_train = X[k][X[k].time < split_date]
            y_station_train = y[k][y[k].index < split_date]
                      
            # Split by date on two groups (train and test), make path_to_save_train and nn_test
            st_path_train = os.path.join(path_to_save_train, k) 
            if not os.path.isdir(st_path_train):
                os.makedirs(st_path_train)
            
            with open(os.path.join(st_path_train, 'objects.npy'),'wb') as f:
                pickle.dump(X_station_train, f)
                # np.save(f, X_station)
            with open(os.path.join(st_path_train, 'target.npy'),'wb') as f:
                pickle.dump(y_station_train, f)
                # np.save(f, y_station)

            X_station_test = X[k][X[k].time >= split_date]
            y_station_test = y[k][y[k].index >= split_date]  

            st_path_test = os.path.join(path_to_save_test, k) 
            if not os.path.isdir(st_path_test):
                os.makedirs(st_path_test)
            
            with open(os.path.join(st_path_test, 'objects.npy'),'wb') as f:
                pickle.dump(X_station_test, f)
                # np.save(f, X_station)
            with open(os.path.join(st_path_test, 'target.npy'),'wb') as f:
                pickle.dump(y_station_test, f)
                # np.save(f, y_station)

        st_split_dict = train_val_test_split(path_to_save_train, train = 0.5, val = 0.25, test = 0.25, verbose = True)
        # path_to_dump = os.path.join('..', 'data', 'nn_train')
        X_train, y_train = extract_splitted_data(path_to_save_train, st_split_dict) # X_train, y_train
        X_test, y_test = extract_splitted_data(path_to_save_test, st_split_dict)
        # X_test, y_test = extract_splitted_data(path_to_save, st_split_dict)  # And below I shouldn't read these files twice
    else:
        # path_to_data = os.path.join('..', 'data', 'nn_train')
        print("Generate dataset skipped, reading")
        st_split_dict = train_val_test_split(path_to_save_train, train = 0.5, val = 0.25, test = 0.25, verbose = True)
        # path_to_dump = os.path.join('..', 'data', 'nn_train')
        X_train, y_train = extract_splitted_data(path_to_save_train, st_split_dict)

        X_test, y_test = extract_splitted_data(path_to_save_test, st_split_dict)
        print("Generate dataset skipped, reading - done")


    #initialize and load model
    print("Initializing NN")
    # batch_size = batch_size
    with open(nn_config_path) as fs:
        args = json.load(fs)

    #os.path.join('..', 'data', 'nn_train')


    # logger = TensorBoardLogger(save_dir='../logs/wind', name='windnet')

    # dm = WindDataModule(X=X, y=y, batch_size=batch_size, downsample=False)
    # model = WindNetPL(args)
    # logger = TensorBoardLogger(save_dir='../logs/wind', name='windnet')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=max_epoch,
                        # accelerator="cpu",
                        gpus=[0],
                        benchmark=True,
                        check_val_every_n_epoch=1,
                        # callbacks=[early_stop_callback]
    )

    dm = WindDataModule(X=X_train, y=y_train, batch_size=batch_size, downsample=False)
    model = WindNetPL(args)

    print("Initializing NN - done")

    print("Training NN")
    print("See, e.g., tensorboard")

    trainer.fit(model, dm)

    print("Training NN - done")

    # st_split_dict = train_val_test_split("/workspace/data/nn_test_2016", train = 0.5, val = 0, test = 0.5, verbose = True)
    # X, y = extract_splitted_data("/workspace/data/nn_test_2016", st_split_dict)
    dm = WindDataModule(X=X_test, y=y_test, batch_size=batch_size, downsample=False)
    print("Testing NN")
    with open('../data/test_metrics.txt', 'w') as f:
        with redirect_stdout(f):
            trainer.test(model, dm)

    
    
    

if __name__ == "__main__":
    # assert len(sys.argv) > 1, "Provide path to config file"
    if len(sys.argv) == 1:
        sys.argv.append('train_conf.json')
    path_to_config = sys.argv[1]
    t1 = time.time()
    infer(path_to_config=path_to_config)
    t2 = time.time()
    print("Total time: ", t2 - t1)
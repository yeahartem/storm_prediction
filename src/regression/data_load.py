import sys,os
sys.path.append(os.getcwd())
from src.utils.data_utils import round_to_closest_indices, make_padding
import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
import yaml
import hashlib
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars
import gc
from functools import partial
import glob 
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def clean_start():
    for f in glob.glob("tmp_t*"):
        os.remove(f)

class DataPreLoader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg    
        logging.info("--- MULTI REGRESSION ---")
        assert (cfg.process.precision == 16 and not cfg.train.normalize) or (cfg.process.precision == 32 and cfg.train.normalize), \
        ''' 16 bit is already normalized. 32 bit is not normalized'''        
        self.generate_hash()       
        self.dataset_torch = self.load_climate_data()
        self.dataset_torch = self.time_crop(self.dataset_torch)

        clean_start()
        if self.cfg.train.make_tmp_target_file:
            if self.data_exists():
                self.load_data()
            else:
                self.prepare_target_df()
                self.target_df_to_array()
                self.save_data()
        else:   
            self.prepare_target_df()
            self.target_df_to_array()
        self.log_data()


    def load_climate_data(self):
        dtype = np.float16 if self.cfg.process.precision == 16 else np.float32
        self.time_coords = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lon.npy'))
        var_data = np.empty(
            (len(self.cfg.process.variables), len(self.time_coords), len(self.lat_coords), len(self.lon_coords)),
            dtype=dtype)
        for i, var in enumerate(self.cfg.process.variables):
            var_data[i] = np.load(os.path.join(self.cfg.train.data_dir, var + f'_{self.cfg.process.precision}.npy'))
        logging.info(f"CMIP data loaded {var_data.shape}")
        var_data, self.shift = make_padding(var_data, self.cfg.half_side_size)
        logging.info(f"Padded data shape {var_data.shape}")
        var_data_torch = torch.from_numpy(var_data).half() if self.cfg.process.precision == 16 else torch.from_numpy(var_data)

        return var_data_torch


    def time_crop(self, var_data):
        start_date = datetime.strptime(self.cfg.train.start_time, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.cfg.train.end_time, '%Y-%m-%d').date()
        start_index = self.time_coords.searchsorted(start_date)
        end_index = self.time_coords.searchsorted(end_date)
        var_data = var_data[:, start_index:end_index, :, :]
        self.time_coords = self.time_coords[start_index:end_index]
        assert len(self.time_coords) == var_data.shape[1]
        logging.info(f"Shape with time limits {var_data.shape}")
        return var_data
    
    #### Target prep

    def quantile_window(self, values, q):           
        if len(values)< self.cfg.train.time_agg_window:
            return None
        else:
            return np.quantile(sliding_window_view(np.array(values), window_shape=self.cfg.train.time_agg_window), q, axis = 1, method='weibull')
            
    def align_time(self, values):
        if len(values)< self.cfg.train.time_agg_window:
            return None
        else:
            values = self.time_coords.searchsorted(values) # time into inds
            values = np.array(values)[self.cfg.train.time_agg_window//2:len(values) - self.cfg.train.time_agg_window//2 + 1]
            return values

    def stations_to_data_grid(self, stations_df: polars.DataFrame) -> polars.DataFrame:
        """ maps stations to the data grid pixels """
        start_time = time.process_time()   
        coords = np.array([*stations_df["station_name"].to_numpy()])
        lat_vector = round_to_closest_indices(coords[:,0], self.lat_coords)
        lon_vector = round_to_closest_indices(coords[:,1], self.lon_coords)
        stations_df = stations_df.with_columns(
                            polars.Series(name="lat", values=lat_vector),
                            polars.Series(name="lon", values=lon_vector)
                            )
        logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")
        return stations_df
    

    def prepare_target_df(self):
        target_df = polars.read_parquet(os.path.join(self.cfg.train.data_dir, self.cfg.train.target_data_file))
        logging.info(f"Target time bounds {target_df['time'].min()}, {target_df['time'].max()}")
        logging.info(f"Data time bounds {self.time_coords.min()}, {self.time_coords.max()}")
        logging.info(f"Records before preparation {len(target_df)}")
        start_date = pd.to_datetime(self.cfg.train.start_time)
        end_date = pd.to_datetime(self.cfg.train.end_time)
        target_df = target_df.filter((polars.col('time') >= start_date) & (polars.col('time') < end_date))
        logging.info(f"Target time bounds {target_df['time'].min()}, {target_df['time'].max()}")
        logging.info(f"Data time bounds {self.time_coords.min()}, {self.time_coords.max()}")
        target_df = (target_df
                    .with_columns(
                                 [polars.concat_list(polars.col('lat'),
                                  polars.col('lon')).alias('station_name')]))
        target_df = target_df.drop("lat", "lon")
        target_df = (target_df
                    .lazy()        
                    .sort("time")
                    .groupby(["station_name"])
                    .agg(
                        [polars.col('time').apply(self.align_time),
                         polars.col('y').apply(partial(self.quantile_window, q=0.96)).alias('y1'),
                         polars.col('y').apply(partial(self.quantile_window, q=0.85)).alias('y2'),
                         polars.col('y').apply(partial(self.quantile_window, q=0.65)).alias('y3'),
                         polars.col('y').apply(partial(self.quantile_window, q=0.45)).alias('y4'),
                         polars.col('y').apply(partial(self.quantile_window, q=0.35)).alias('y5'),
                         polars.col('y').apply(partial(self.quantile_window, q=0.10)).alias('y6'),])
                    .collect())
        
        target_df = self.stations_to_data_grid(stations_df=target_df)
        target_df = target_df.drop("station_name")
        self.target_df = target_df.drop_nulls()
        logging.info(f"Stations before droppping: {len(target_df)}")

    @staticmethod
    def split_target(y, dates_train, clipped_dates):
        y_train = y[:len(dates_train)].astype(np.int16)
        y_test = y[len(dates_train):len(clipped_dates)].astype(np.int16)
        return y_train, y_test

    def target_df_to_array(self):
        split_date = datetime.strptime(self.cfg.train.start_of_test, '%Y-%m-%d').date()
        split_index = self.time_coords.searchsorted(split_date)
        train_data_idxs = []
        test_data_idxs = []
        stations = []
        for dates, y1, y2, y3, y4, y5, y6, lat, lon in self.target_df.rows():
            if (lat is not None) and (lon is not None) and (dates is not None) and (y1 is not None):
                if isinstance(dates, float):
                    continue
                assert len(dates) == len(y1), f'dates axis: {len(dates)} target axis: {len(y1)}'
                stations.append([lat, lon])
                clipped_dates = dates[dates<(self.time_coords.shape[0]-self.cfg.time_window - 1)]
                clipped_dates = clipped_dates[clipped_dates>self.cfg.time_window]
                dates_train = clipped_dates[clipped_dates < split_index]
                dates_test = clipped_dates[clipped_dates >= split_index]
                
                if isinstance(dates_test, float):
                    continue
                if (len(clipped_dates) - len(dates_test)) < 2:
                    continue
                
                target_q_list_train = []
                target_q_list_test = []

                for y in [y1, y2, y3, y4, y5, y6]:
                    y_train, y_test = self.split_target(y, dates_train, clipped_dates)
                    target_q_list_train.append(y_train)
                    target_q_list_test.append(y_test)

                target_q_array_train = np.stack(target_q_list_train)
                target_q_array_test = np.stack(target_q_list_test)

                assert len(dates_train) == len(y_train), f"train len dates {dates_train.shape} len labels {y_train.shape}"
                assert len(dates_test) == len(y_test), f"test len dates {dates_test.shape} len labels {y_test.shape}"
                
                arr_train = np.stack([np.full(len(dates_train), lat, dtype=np.int16), np.full(len(dates_train), lon, dtype=np.int16), dates_train])
                arr_test = np.stack([np.full(len(dates_test), lat, dtype=np.int16), np.full(len(dates_test), lon, dtype=np.int16), dates_test])

                arr_train = np.concatenate((arr_train, target_q_array_train), axis=0)
                arr_test = np.concatenate((arr_test, target_q_array_test), axis=0)

                train_data_idxs.append(arr_train)
                test_data_idxs.append(arr_test)

        self.train_data_idxs = np.concatenate(train_data_idxs, axis=1)
        self.test_data_idxs = np.concatenate(test_data_idxs, axis=1)
        self.train_data_idxs = self.train_data_idxs[:, ::self.cfg.train.time_freq]
        self.test_data_idxs = self.test_data_idxs[:, ::self.cfg.train.time_freq]
        #index shift due to padding
        self.train_data_idxs[0, :] += self.shift[0] #lat
        self.train_data_idxs[1, :] += self.shift[1] #lon
        self.test_data_idxs[0, :] += self.shift[0]  #lat
        self.test_data_idxs[1, :] += self.shift[1]  #lon

        logging.info(f'TRAIN MIN LAT {self.test_data_idxs[0, :].min()} LON {self.test_data_idxs[1, :].min()}')
        logging.info(f'TRAIN MAX LAT {self.test_data_idxs[0, :].max()} LON {self.test_data_idxs[1, :].max()}')
        logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared test {self.test_data_idxs.shape[1]}')
        gc.collect()

    ### Utils for preload
    def generate_hash(self):
        config_str = yaml.dump(OmegaConf.to_yaml(self.cfg), sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    def data_exists(self):
        return os.path.isfile(f'tmp_Q_train_{self.config_hash}.npz')

    def save_data(self):
        np.savez_compressed(f'tmp_Q_train_{self.config_hash}.npz', self.train_data_idxs)
        np.savez_compressed(f'tmp_Q_test_{self.config_hash}.npz', self.test_data_idxs)

    def load_data(self):
        self.train_data_idxs = np.load(f'tmp_Q_train_{self.config_hash}.npz')['arr_0']
        self.test_data_idxs = np.load(f'tmp_Q_test_{self.config_hash}.npz')['arr_0']

    def log_data(self):
        logging.info(f"Train size: {self.train_data_idxs.shape[1]}, test size: {self.test_data_idxs.shape[1]}")
        logging.info(f"Target min: {self.train_data_idxs[3, :].min()}, target max: {self.train_data_idxs[3, :].max()}")
        logging.info(f"Target mean: {self.train_data_idxs[3, :].mean()}, target std: {self.train_data_idxs[3, :].std()}")
        logging.info(f"Balance train: {self.get_class_balance(self.train_data_idxs[3, :])}, balance test:{self.get_class_balance(self.test_data_idxs[3, :])}")
        for i, var in enumerate(self.cfg.process.variables):
            logging.info(f"{var} mean: {self.dataset_torch[i].mean()}, std: {self.dataset_torch[i].std()}")

    def get_class_balance(self, target_array):
        positive = np.sum(target_array >= self.cfg.train.target_threshold)
        all = target_array.shape[0]
        return positive/all



    
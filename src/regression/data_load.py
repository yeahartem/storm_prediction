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

    def align_time(self, target_df):
        start_time = time.process_time()   
        dates = target_df["time"].to_numpy()
        values = self.time_coords.searchsorted(dates) # time into inds
        target_df = target_df.with_columns(
                            polars.Series(name="time", values=values),
                            )
        logging.info(f"Time align took {time.process_time() - start_time} seconds")
        return target_df

    def stations_to_data_grid(self, stations_df: polars.DataFrame) -> polars.DataFrame:
        """ maps stations to the data grid pixels """
        start_time = time.process_time()   
        coords = np.array([*stations_df["station_name"].to_numpy()])
        lat = coords[:, 0]
        lon = coords[:, 1]
        lat_vector = round_to_closest_indices(lat, self.lat_coords)
        lon_vector = round_to_closest_indices(lon, self.lon_coords)
        stations_df = stations_df.with_columns(
                            [polars.concat_list(
                             polars.Series(name="lat", values=lat_vector),
                             polars.Series(name="lon", values=lon_vector)).alias('station_name')
                            ])
        logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")
        return stations_df
    

    def prepare_target_df(self):
        target_df = polars.read_parquet(os.path.join(self.cfg.train.data_dir, self.cfg.train.target_data_file))
        logging.info(f"Records before preparation {len(target_df)}")
        start_date = pd.to_datetime(self.cfg.train.start_time)
        end_date = pd.to_datetime(self.cfg.train.end_time)
        target_df = target_df.filter((polars.col('time') >= start_date) & (polars.col('time') < end_date))
        logging.info(f"Target time bounds {target_df['time'].min()}, {target_df['time'].max()}")
        logging.info(f"Data time bounds {self.time_coords.min()}, {self.time_coords.max()}")
        logging.info(f"Stations before aggregation: {target_df.n_unique(subset=['lat', 'lon'])}")
        target_df = self.align_time(target_df)
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
                        [polars.col('time'),
                         polars.col('y'),
                        ])
                    .collect())
        
        target_df = self.stations_to_data_grid(target_df)
        target_df = target_df.unique(subset=['station_name'])
        self.target_df = target_df.drop_nulls()
        logging.info(f"Stations after aggregation: {len(target_df)}")


    def target_df_to_array(self):
        split_date = datetime.strptime(self.cfg.train.start_of_test, '%Y-%m-%d').date()
        split_index = self.time_coords.searchsorted(split_date)
        targets_list = []

        start_time = time.process_time()   
        for target_df_row in self.target_df.rows():
            coords, dates, y = target_df_row
            if len(y) < max(self.cfg.train.time_agg_window, self.cfg.time_window):
                continue
            targets_list.append(self.pixel_aggregation(coords, dates, y))
        logging.info(f"Pixel loop took {time.process_time() - start_time} seconds")

        target_array = np.concatenate(targets_list, axis=1)
        target_array = target_array.astype(np.int32)
        del targets_list
        target_array = target_array[:, ::self.cfg.train.time_freq]
        target_array[0, :] += self.shift[0] #lat
        target_array[1, :] += self.shift[1] #lon

        self.train_data_idxs = target_array[:, target_array[2, :] < split_index]
        print(f"train {self.train_data_idxs.shape}")
        self.test_data_idxs = target_array[:, target_array[2, :] > split_index]
        print(f"test1 {self.test_data_idxs.shape}")

        self.test_data_idxs = target_array[:, target_array[2, :] < len(self.time_coords)]
        print(f"test2 {self.test_data_idxs.shape}")

        logging.info(f'TRAIN MIN LAT {self.test_data_idxs[0, :].min()} LON {self.test_data_idxs[1, :].min()}')
        logging.info(f'TRAIN MAX LAT {self.test_data_idxs[0, :].max()} LON {self.test_data_idxs[1, :].max()}')
        logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared test {self.test_data_idxs.shape[1]}')
        gc.collect()


    def pixel_aggregation(self, coords, dates, y):
        # assert len(dates) == len(y), f'dates axis: {len(dates)} target axis: {len(y)}'
        lat, lon = coords
        #df = pd.DataFrame(data={'dates': dates, 'y': y}).sort_values(by=['dates'])
        #pixel = df.groupby(['dates']).agg(lambda x: np.percentile(x, q=0.95, method='weibull')) # in pixel aggregation
        #dates = pixel.index.to_numpy()
        #y = np.squeeze(pixel.values)
        y = np.array(y)
        # dates = np.array(list(map(int, dates)))
        dates = np.array(dates)

        # aggregate target with given time_agg_window 
        y_agg_quantlies = np.quantile(sliding_window_view(y, window_shape=self.cfg.train.time_agg_window), 
                        q=[0.96, 0.85, 0.70, 0.50, 0.25, 0.15, 0.05],
                        axis = 1,
                        method='weibull')
        
        i = 1 if self.cfg.train.time_agg_window % 2 == 0 else 0
        if self.cfg.time_window > self.cfg.train.time_agg_window:
            # clip dates according to time_window
            dates_clipped = dates[self.cfg.time_window//2:
                                  len(dates)-self.cfg.time_window//2 + i] 
            
            y_agg_quantlies = y_agg_quantlies[self.cfg.time_window-self.cfg.train.time_agg_window:
                                              len(y_agg_quantlies) + self.cfg.train.time_agg_window - self.cfg.time_window - 1]
        else:
            dates_clipped = dates[self.cfg.train.time_agg_window//2:
                                  len(dates)-self.cfg.train.time_agg_window//2 + i] 
             # y_agg_quantlies not changed
        
        # assert len(dates_clipped) == len(y_agg_quantlies[1]), f'd {len(dates_clipped)} y {y_agg_quantlies.shape[1]}'
        target_array = np.stack([np.full(len(dates_clipped), lat, dtype=np.int32),
                                 np.full(len(dates_clipped), lon, dtype=np.int32),
                                 dates_clipped])
        target_array = np.concatenate((target_array, y_agg_quantlies), axis=0)
        return target_array
    
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



    

import sys,os
sys.path.append(os.getcwd())
import logging
import numpy as np
import torchvision  
from omegaconf import DictConfig, OmegaConf
import yaml
import hashlib
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars
import gc


class DataPreLoader:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg    
        self.generate_hash()       
        self.time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
        if self.data_exists():
            self.load_data()
        else:
            self.prepare_target()
            self.save_data()
        self.dataset_as_blocks = self.load_climate_data()

        if self.cfg.normalize:
            mean_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            std_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        else: self.transform = None


    def load_climate_data(self):

        start_time = time.process_time()
        time_coords = np.load(os.path.join(self.cfg.data_dir, 'time.npy')).astype('datetime64[D]')
        lat_coords = np.load(os.path.join(self.cfg.data_dir, 'lat.npy'))
        lon_coords = np.load(os.path.join(self.cfg.data_dir, 'lon.npy'))
        dtype = np.float16 if self.cfg.precision == 16 else np.float32
        var_data = np.empty((len(self.cfg.variables), len(time_coords), len(lat_coords), len(lon_coords)), dtype=dtype)

        for i, var in enumerate(self.cfg.variables):
            var_data[i] = np.load(os.path.join(self.cfg.data_dir, var + f'_{self.cfg.precision}.npy'))

        var_data = np.moveaxis(var_data, 0, 1)
        var_data_windows = np.lib.stride_tricks.sliding_window_view(var_data,
                                                        (self.cfg.time_window, var_data.shape[1], 2 * self.cfg.half_side_size + 1,
                                                            2 * self.cfg.half_side_size + 1))
        var_data_windows = np.moveaxis(np.squeeze(var_data_windows), 0, 2)
        logging.info(f"Numpy block preparation took {time.process_time() - start_time} seconds")

        return var_data_windows


    def max_window(self, values):            
        if len(values)< self.cfg.time_window:
            return None
        else:
            return np.max(sliding_window_view(np.array(values), window_shape = self.cfg.time_window), axis = 1)
            

    def align_time(self, values):
        if len(values)< self.cfg.time_window:
            return None
        else:
            values = self.time_coords.searchsorted(values) # time into inds
            return np.array(values)[self.cfg.time_window//2:len(values) - self.cfg.time_window//2]


    def align_coords(self, values):
        lat, lon = values - self.cfg.half_side_size 
        if (lat < 0) or (lon < 0):
            return None
        elif (lat > (len(self.lat_coords) - 2*self.cfg.half_side_size - 1)) or (lon > (len(self.lon_coords) - 2*self.cfg.half_side_size - 1)):
            return None
        else:
            return np.array((lat, lon))


    def prepare_target(self):

        logging.info('tmp file not found, processing')            
        start_time = time.process_time()  
        target_df = polars.read_parquet(self.cfg.path_to_prepared_target_data)
        logging.info(f"Records before preparation {len(target_df)}")
        target_df = (
            target_df
            .lazy()        
            .sort("time")
            .groupby(["station_name"])
            .agg(
                [polars.col('time').apply(self.align_time), polars.col('y').apply(self.max_window)]
            )
            .collect()
        )
        target_df = target_df.with_columns(polars.col('station_name').apply(self.align_coords).keep_name())
        logging.info(f"Stations before droppping: {len(target_df)}")
        logging.info(f"Time to prepare target {time.process_time() - start_time} seconds")
        split_date = datetime.strptime(self.cfg.start_of_test, '%Y-%m-%d').date()
        split_index = self.time_coords.searchsorted(split_date)

        train_data_idxs = []
        test_data_idxs = []
        stations = []

        for coords, dates, y in target_df.rows(): # ugly, will be fixed
            if (coords is not None) and (dates is not None) and (y is not None):
                if not (any(np.isnan(np.array(coords), casting='unsafe')) and any(np.isnan(np.array(dates), casting='unsafe')) and any(np.isnan(np.array(y), casting='unsafe'))):

                    stations.append(coords)
                    clipped_dates = dates[dates < (self.time_coords.shape[0] - self.cfg.time_window - 1)]
                    dates_train = clipped_dates[clipped_dates < split_index]
                    dates_test = clipped_dates[clipped_dates >= split_index]

                    y_train = y[:len(dates_train)]
                    y_test = y[len(dates_train):len(clipped_dates)]
                    arr_train = np.stack([np.full(len(dates_train),coords[0], dtype=np.int16), np.full(len(dates_train), coords[1], dtype=np.int16), dates_train, y_train])
                    arr_test = np.stack([np.full(len(dates_test),coords[0], dtype=np.int16), np.full(len(dates_test), coords[1], dtype=np.int16), dates_test, y_test])

                    train_data_idxs.append(arr_train)
                    test_data_idxs.append(arr_test)
        
        self.stations = np.array(stations)
        self.train_data_idxs = np.concatenate(train_data_idxs, axis=1)
        self.test_data_idxs = np.concatenate(test_data_idxs, axis=1)
        logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared test {self.test_data_idxs.shape[1]}')
        gc.collect()

    def generate_hash(self):
        config_str = yaml.dump(OmegaConf.to_yaml(self.cfg), sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    def data_exists(self):
        return os.path.isfile(f'tmp_train_{self.config_hash}.npz')

    def save_data(self):
        np.savez_compressed(f'tmp_train_{self.config_hash}.npz', self.train_data_idxs)
        np.savez_compressed(f'tmp_test_{self.config_hash}.npz', self.test_data_idxs)

    def load_data(self):
        self.train_data_idxs = np.load(f'tmp_train_{self.config_hash}.npz')['arr_0']
        self.test_data_idxs = np.load(f'tmp_test_{self.config_hash}.npz')['arr_0']


    def log_data(self):

        logging.info(f"Train size: {self.train_data_idxs.shape[1]}, test size: {self.test_data_idxs.shape[1]}")
        logging.info(f"Station count: {len(self.station)}")

        # logging.info(f"Train rectangle: {dm.result_train_rectangle}")
        # logging.info(f"Test rectangle: {dm.result_test_rectangle}")
        # logging.info(f"Extreme stations train: {dm.extreme_stations_train}")
        # logging.info(f"Extreme stations test: {dm.extreme_stations_test}")

        logging.info(f"Target min: {self.train_data_idxs[:,3].min()}, target max: {self.train_data_idxs[:,3].max()}")
        logging.info(f"Target mean: {self.train_data_idxs[:,3].mean()}, target std: {self.train_data_idxs[:,3].std()}")

        # logging.info(f"Data min: {self.dataset_as_blocks.min(axis=(1,2,3))}, data max: {self.dataset_as_blocks.max(axis=(1,2,3))}")
        # logging.info(f"Data mean: {self.dataset_as_blocks.mean(axis=(1,2,3))}, data std: {self.dataset_as_blocks.std(axis=(1,2,3))}")

class DataInferPreLoader(DataPreLoader):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg    
        self.generate_hash()       
        lat_min, lat_max, lon_min, lon_max = cfg.test_coords['lat_min'], cfg.test_coords['lat_max'], cfg.test_coords['lon_min'], cfg.test_coords['lon_max']
        self.time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
        self.lat_idxs = np.where(np.logical_and((self.lat_coords <= lat_max), (self.lat_coords >= lat_min)))
        self.lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
        self.lon_idxs = np.where(np.logical_and((self.lon_coords <= lon_max), (self.lon_coords >= lon_min)))
        
        if self.data_exists():
            self.load_data()

        self.dataset_as_blocks = self.load_climate_data()
        # lat_idxs = np.arange(self.dataset_as_blocks.shape[0])
        # lon_idxs = np.arange(self.dataset_as_blocks.shape[1])
        tim_idxs = np.arange(self.dataset_as_blocks.shape[2])
        idxs_mesh = np.meshgrid(self.lat_idxs, self.lon_idxs, tim_idxs)
        self.train_data_idxs = None
        self.test_data_idxs = np.vstack((idxs_mesh[0].flatten(), idxs_mesh[1].flatten(), idxs_mesh[2].flatten(), np.empty(len(idxs_mesh[2].flatten())))) # walk order: time -> lat -> lon (2 -> 0 -> 1)
        # self.test_data_idxs = np.vstack((self.test_data_idxs, (-1) * np.ones(self.test_data_idxs[-1], dtype=int)))
        # self.train_data_idxs = np.concatenate(train_data_idxs, axis=1)
        # self.test_data_idxs = np.concatenate(test_data_idxs, axis=1)
        # logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared infer {self.test_data_idxs.shape[1]}')

        if self.cfg.normalize:
            mean_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            std_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        else: self.transform = None
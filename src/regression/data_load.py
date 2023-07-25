import sys,os
sys.path.append(os.getcwd())
import logging
import numpy as np
import torch
import torchvision  
from omegaconf import DictConfig, OmegaConf
import yaml
import hashlib
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars
import gc



class DataPreLoaderAlt:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg    
        assert (cfg.precision == 16 and not cfg.normalize) or (cfg.precision == 32 and cfg.normalize), \
        ''' 16 bit is already normalized. 32 bit is not normalized'''        
        self.generate_hash()       
        self.time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
        self.dataset_torch = self.load_climate_data()
        if self.data_exists():
            self.load_data()
        else:
            self.prepare_target()
            self.save_data()
        

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

        #map padding
        var_data_padded = self.make_padding(var_data)

        var_data_torch = torch.from_numpy(var_data_padded).half() if self.cfg.precision == 16 else torch.from_numpy(var_data_padded)
        logging.info(f"Climate data preparation took {time.process_time() - start_time} seconds")
        return var_data_torch


    def max_window(self, values):           
        if len(values)< self.cfg.time_agg_window:
            return None
        else:
            return np.max(sliding_window_view(np.array(values), window_shape = self.cfg.time_agg_window), axis = 1)
            
    def align_time(self, values):
        if len(values)< self.cfg.time_window:
            return None
        else:
            values = self.time_coords.searchsorted(values) # time into inds
            values = np.array(values)[self.cfg.time_window//2 + 1:len(values) - self.cfg.time_window//2 - 1]
            return values


    def stations_to_data_grid(self, stations_df: polars.DataFrame) -> polars.DataFrame:
        """ maps stations to the data grid pixels """
        start_time = time.process_time()   
        coords = np.array([*stations_df["station_name"].to_numpy()])
        lat_vector = self.round_to_closest_indices(coords[:,0], self.lat_coords)
        lon_vector = self.round_to_closest_indices(coords[:,1], self.lon_coords)
        stations_df = stations_df.with_columns(
                            polars.Series(name="lat", values=lat_vector),
                            polars.Series(name="lon", values=lon_vector)
                            )
        logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")
        return stations_df

    def make_padding(self, data):
        quadrants = self.extract_quadrants(data)
        padded_map = self.assemble_padded_map(quadrants)
        return padded_map
    
    def extract_quadrants(self, data):
        halfs = {"lat": data.shape[-2] // 2, "lon": data.shape[-1] // 2}
        first_quadrant  = data[..., halfs['lat']:, halfs['lon']:]
        second_quadrant = data[..., halfs['lat']:, :halfs['lon']]
        third_quadrant  = data[..., :halfs['lat'], :halfs['lon']]
        fourth_quadrant = data[..., :halfs['lat'], halfs['lon']:]
        self.fourth_q_shape = (fourth_quadrant.shape[-2], fourth_quadrant.shape[-1])
        return first_quadrant, second_quadrant, third_quadrant, fourth_quadrant

    def assemble_padded_map(self, quadrants):
        try:
            q_flipped = [q.reindex(lat=list(reversed(q.lat))) for q in quadrants]
        except AttributeError:
            q_flipped = [np.flip(q, axis=-2) for q in quadrants]
        
        column_0 = np.concatenate([q_flipped[2], quadrants[3], quadrants[0], q_flipped[1]], axis=-2)
        column_1 = np.concatenate([q_flipped[3], quadrants[2], quadrants[1], q_flipped[0]], axis=-2)
        #0 1 0 1

        padded_map = np.concatenate((column_0, column_1, column_0, column_1), axis=-1)

        return padded_map

    @staticmethod
    def round_to_closest_indices(arr, values):
        values = np.array(values)
        indices = np.searchsorted(values, arr)
        indices = np.clip(indices, 1, len(values) - 1)
        left_values = values[indices - 1]
        right_values = values[indices]
        left_indices = indices - 1
        right_indices = indices
        closest_indices = np.where(np.abs(arr - left_values) <= np.abs(arr - right_values), left_indices, right_indices)
        return closest_indices
    

    @staticmethod
    def round_to_closest_values(arr, values):
        values = np.array(values)
        indices = np.searchsorted(values, arr)
        indices = np.clip(indices, 1, len(values) - 1)
        left_values = values[indices - 1]
        right_values = values[indices]
        closest_values = np.where(np.abs(arr - left_values) <= np.abs(arr - right_values), left_values, right_values)
        return closest_values
    

    def prepare_target(self):
        logging.info('tmp file not found, processing')            
        start_time = time.process_time()  
        target_df = polars.read_parquet(self.cfg.path_to_prepared_target_data)
        logging.info(f"Records before preparation {len(target_df)}")
        target_df = target_df.with_columns([polars.concat_list(polars.col('lat'),
                                                               polars.col('lon')).alias('station_name')])
        target_df = target_df.drop("lat", "lon")
        target_df =( 
            target_df
            .lazy()        
            .sort("time")
            .groupby(["station_name"])
            .agg(
                [polars.col('time').apply(self.align_time), polars.col('y').apply(self.max_window)]
            )
        )
        target_df = target_df.collect()
        target_df = self.stations_to_data_grid(stations_df=target_df)
        target_df = target_df.drop("station_name")
        # filter out stations that are too close to the edge
        # target_df = target_df.filter((polars.col("lat") > self.cfg.half_side_size) &
        #                              (polars.col("lon") > self.cfg.half_side_size) &
        #                              (polars.col("lat") < (len(self.lat_coords) - self.cfg.half_side_size - 1)) &
        #                              (polars.col("lon") < (len(self.lon_coords) - self.cfg.half_side_size- 1))
        #                             )
        
        logging.info(f"Stations before droppping: {len(target_df)}")
        logging.info(f"Time to prepare target {time.process_time() - start_time} seconds")
        split_date = datetime.strptime(self.cfg.start_of_test, '%Y-%m-%d').date()
        split_index = self.time_coords.searchsorted(split_date)

        train_data_idxs = []
        test_data_idxs = []
        stations = []
        target_df = target_df.drop_nulls()
        for dates, y, lat, lon in target_df.rows():
            if (lat is not None) and (lon is not None) and (dates is not None) and (y is not None):
                if not (any(np.isnan(np.array([lat, lon]), casting='unsafe')) and any(np.isnan(np.array(dates), casting='unsafe')) and any(np.isnan(np.array(y), casting='unsafe'))):

                    stations.append([lat, lon])
                    clipped_dates = dates[dates<(self.time_coords.shape[0]-self.cfg.time_window - 1)]
                    clipped_dates = clipped_dates[clipped_dates>self.cfg.time_window]
                    dates_train = clipped_dates[clipped_dates < split_index]
                    dates_test = clipped_dates[clipped_dates >= split_index]

                    y_train = y[:len(dates_train)]
                    y_test = y[len(dates_train):len(clipped_dates)]
                    arr_train = np.stack([np.full(len(dates_train), lat, dtype=np.int16), np.full(len(dates_train), lon, dtype=np.int16), dates_train, y_train])
                    arr_test = np.stack([np.full(len(dates_test), lat, dtype=np.int16), np.full(len(dates_test), lon, dtype=np.int16), dates_test, y_test])

                    train_data_idxs.append(arr_train)
                    test_data_idxs.append(arr_test)

        self.stations = np.array(stations)
        self.train_data_idxs = np.concatenate(train_data_idxs, axis=1)
        self.test_data_idxs = np.concatenate(test_data_idxs, axis=1)
        #index shift due to padding
        self.train_data_idxs[0, :] += self.fourth_q_shape[0] #lat
        self.train_data_idxs[1, :] += self.fourth_q_shape[1] #lon
        self.test_data_idxs[0, :] += self.fourth_q_shape[0]  #lat
        self.test_data_idxs[1, :] += self.fourth_q_shape[1]  #lon
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
        logging.info(f"Target min: {self.train_data_idxs[:,3].min()}, target max: {self.train_data_idxs[:,3].max()}")
        logging.info(f"Target mean: {self.train_data_idxs[:,3].mean()}, target std: {self.train_data_idxs[:,3].std()}")
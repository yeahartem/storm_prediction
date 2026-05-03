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
import pickle


@rank_zero_only 
def clean_start():
    """ограничивает выполнение функции только “нулевым” процессом в распределённом запуске и удаляет временные файлы по маске (я не нашел у нас tmp файлы)"""
    for f in glob.glob("tmp_Q_t*"):
        os.remove(f)



class DataPreLoader:
    def __init__(self, cfg: DictConfig): # self это dm.DPL
        self.cfg = cfg    
        logging.info("--- MULTI REGRESSION ---")
        assert (cfg.process.precision == 16 and not cfg.train.normalize) or (cfg.process.precision == 32 and cfg.train.normalize), \
        ''' 16 bit is already normalized. 32 bit is not normalized'''        
        self.generate_hash()       
        self.dataset_torch = self.load_climate_data() # понять зачем в float32 переводят? Если в обучении будет падать из-за памяти, перевести в 16
        self.dataset_torch = self.time_crop(self.dataset_torch)

        if self.cfg.train.use_elevation:
            self.load_elevation_data()

        if self.cfg.train.get('synthetic_labels', False):
            self._load_sfcwindmax_for_synthetic()
            self.station_threshold_lookup = {}  # use global synthetic threshold for all cells
        else:
            self.station_threshold_lookup = self.load_station_thresholds()

        clean_start() # ограничивает выполнение функции только “нулевым” процессом в распределённом запуске и удаляет временные файлы по маске

        if self.cfg.train.make_tmp_target_file:
            if self.data_exists():
                self.load_data()
            else:
                self.prepare_target_df()
                self.target_df_to_array()
                self.save_data()
        else:   
            self.prepare_target_df()
            self.target_df_to_array() # Переводит в индексы и в форму, чтобы таргет y был на индексе 7
        self.log_data()


    def load_climate_data(self):
        '''
        Подгружаются обработанные CMIP
        '''
        dtype = np.float16 if self.cfg.process.precision == 16 else np.float32
        self.time_coords = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lon.npy'))

        var_data = np.empty(
            (len(self.cfg.train.variables), len(self.time_coords), len(self.lat_coords), len(self.lon_coords)),
            dtype=dtype)
        for i, var in enumerate(self.cfg.train.variables):
            var_data[i] = np.load(os.path.join(self.cfg.train.data_dir, var + f'_{self.cfg.process.precision}.npy'))
        logging.info(f"CMIP data loaded {var_data.shape}")

        if self.cfg.train.spatial_crop: # пока false сделали, так как в preprocess.py сделали кроп 07.02.26
            var_data = self.spatial_crop(var_data)
            logging.info(f"Cropped data shape {var_data.shape}")
        else:
            var_data, self.shift = make_padding(var_data, self.cfg.half_side_size) # Возвращает padded_map, (half_side_size, half_side_size)
            logging.info(f"Padded data shape {var_data.shape}")

        # var_data_torch = torch.from_numpy(var_data).half() if self.cfg.process.precision == 16 else torch.from_numpy(var_data)
        var_data_torch = torch.from_numpy(var_data).type(torch.float32) # Сомнительное дело, понять зачем в 32 переводят?

        return var_data_torch

    def spatial_crop(self, var_data): 
        half_side = self.cfg.half_side_size
        self.lat_min_idx = np.searchsorted(self.lat_coords, self.cfg.train.lat_min)
        self.lat_max_idx = np.searchsorted(self.lat_coords, self.cfg.train.lat_max)
        self.lon_min_idx = np.searchsorted(self.lon_coords, self.cfg.train.lon_min)
        self.lon_max_idx = np.searchsorted(self.lon_coords, self.cfg.train.lon_max)
        
        self.lat_coords_crop = self.lat_coords[self.lat_min_idx: self.lat_max_idx]
        self.lon_coords_crop = self.lon_coords[self.lon_min_idx: self.lon_max_idx]
        logging.info(f"Lat : {min(self.lat_coords_crop)} - {max(self.lat_coords_crop)}, len {len(self.lat_coords_crop)}")
        logging.info(f"Lon: {min(self.lon_coords_crop)} - {max(self.lon_coords_crop)}, len {len(self.lon_coords_crop)}")
        logging.info(f"Lat indexes: {(self.lat_min_idx)} - {(self.lat_max_idx)}, len {len(self.lat_coords_crop)}")
        logging.info(f"Lon indexes: {(self.lon_min_idx)} - {(self.lon_max_idx)}, len {len(self.lon_coords_crop)}")
        # Пути к файлам кэша из конфига
        pad_data_path = self.cfg.train.cache.pad_crop_data_path
        shift_path = self.cfg.train.cache.shift_data_path        
        # Проверяем, нужно ли использовать кэш и существуют ли файлы
        if self.cfg.train.cache.use_cached_padding and os.path.exists(pad_data_path) and os.path.exists(shift_path):
            logging.info(f"Загрузка обработанных данных из кэша...")
            logging.info(f"Файл данных: {pad_data_path}")
            logging.info(f"Файл сдвига: {shift_path}")
            
            # Загружаем данные
            var_data = np.load(pad_data_path)
            with open(shift_path, 'rb') as f:
                self.shift = pickle.load(f)        

        else:
            # Если кэш не используется или файлы не найдены, выполняем тяжелую операцию
            logging.info("Кэш не используется или не найден. Выполняется тяжелая операция padding + crop...")
                
            if  (self.lat_min_idx < half_side) or \
                (self.lon_min_idx < half_side) or \
                (len(self.lon_coords) - self.lon_max_idx < half_side) or \
                (len(self.lat_coords) - self.lat_max_idx < half_side):
                logging.info(f"Pad + crop")

                var_data, self.shift = make_padding(var_data, self.cfg.half_side_size)
                var_data = var_data[
                                    :,
                                    :,
                                    self.lat_min_idx: self.lat_max_idx + 2*half_side + 1,
                                    self.lon_min_idx: self.lon_max_idx + 2*half_side + 1
                                    ]
                
                # --- СОХРАНЯЕМ РЕЗУЛЬТАТЫ В КЭШ ---
                logging.info(f"Сохранение результатов в кэш для будущего использования...")
                # Создаем директорию, если её нет
                os.makedirs(os.path.dirname(pad_data_path), exist_ok=True)
                
                # Сохраняем основной массив в формате NetCDF
                np.save(pad_data_path, var_data)
                logging.info(f"Данные сохранены в: {pad_data_path}")
                
                # Сохраняем объект shift с помощью pickle
                with open(shift_path, 'wb') as f:
                    pickle.dump(self.shift, f)
                logging.info(f"Сдвиг сохранен в: {shift_path}")  
                
            else:
                logging.info(f"Just crop")
                var_data = var_data[
                                    :,
                                    :,
                                    self.lat_min_idx - half_side: self.lat_max_idx + half_side + 1,
                                    self.lon_min_idx - half_side: self.lon_max_idx + half_side + 1
                                    ]
        return var_data

    def time_crop(self, var_data):
        logging.info(f"Shape before time limits in time_crop() in data_load.py: {var_data.shape}")
        start_date = datetime.strptime(self.cfg.train.start_time, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.cfg.train.end_time, '%Y-%m-%d').date()
        start_index = self.time_coords.searchsorted(start_date)
        end_index = self.time_coords.searchsorted(end_date)
        var_data = var_data[:, start_index:end_index, :, :]
        self.time_coords = self.time_coords[start_index:end_index]
        assert len(self.time_coords) == var_data.shape[1]
        logging.info(f"Shape with time limits: {var_data.shape}")
        return var_data
    

    def load_elevation_data(self):
        """Load regridded elevation (same CMIP6 grid) and apply same padding as climate data."""
        elev = np.load(os.path.join(self.cfg.train.data_dir, f'elev_{self.cfg.process.precision}.npy')).astype(np.float32)
        # elev shape: (lat, lon) – same grid as CMIP6 lat.npy / lon.npy
        pad = self.cfg.half_side_size
        elev_padded = np.pad(elev, ((pad, pad), (pad, pad)), mode='edge')
        self.elevation_torch = torch.from_numpy(elev_padded).type(torch.float32)
        logging.info(f"Elevation loaded: raw {elev.shape}, padded {elev_padded.shape}")

    def _load_sfcwindmax_for_synthetic(self):
        """Load sfcWindmax separately for CMIP6 pseudo-label generation.

        Used when synthetic_labels=True and sfcWindmax is not in training variables.
        Applies the same time crop and padding as the main climate data.
        De-normalizes to physical m/s so that synthetic_threshold stays in interpretable units.
        """
        from src.utils.norm_values import mean_channels_cmip6, std_channels_cmip6
        orig_time = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        start_date = datetime.strptime(self.cfg.train.start_time, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.cfg.train.end_time, '%Y-%m-%d').date()
        t0 = int(orig_time.searchsorted(start_date))
        t1 = int(orig_time.searchsorted(end_date))
        path = os.path.join(self.cfg.train.data_dir, f'sfcWindmax_{self.cfg.process.precision}.npy')
        data = np.load(path)[t0:t1].astype(np.float32)  # (time, lat, lon), z-score normalized
        sfc_mean = float(mean_channels_cmip6[0])  # 8.934635 m/s
        sfc_std  = float(std_channels_cmip6[0])   # 4.7078495 m/s
        data = data * sfc_std + sfc_mean           # de-normalize to physical m/s
        pad = self.cfg.half_side_size
        data_padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode='edge')
        self.sfcwindmax_for_labels = torch.from_numpy(data_padded)
        logging.info(f"sfcWindmax for synthetic labels loaded: {data_padded.shape}, physical range [{data.min():.2f}, {data.max():.2f}] m/s")

    def load_station_thresholds(self):
        """Load per-station p95 thresholds and build lookup (lat_idx, lon_idx) -> effective_threshold.
        effective_threshold = max(p95_station, abs_threshold) so the condition becomes:
            positive = (y >= effective_threshold)
        Falls back to cfg.train.target_threshold if file not found.
        """
        thresh_path = os.path.join(self.cfg.train.data_dir, 'station_thresholds.parquet')
        abs_thresh = float(self.cfg.train.get('abs_wind_threshold', 15.0))
        if not os.path.exists(thresh_path):
            logging.info(f"station_thresholds.parquet not found, using global threshold {abs_thresh}")
            return {}
        df = polars.read_parquet(thresh_path)
        # Map lat/lon degrees to grid indices (same as stations_to_data_grid)
        lats = df['lat'].to_numpy()
        lons = df['lon'].to_numpy()
        p95s = df['p95'].to_numpy()
        lat_idxs = round_to_closest_indices(lats, self.lat_coords)
        lon_idxs = round_to_closest_indices(lons, self.lon_coords)
        lookup = {}
        for lat_idx, lon_idx, p95 in zip(lat_idxs, lon_idxs, p95s):
            effective = float(max(p95, abs_thresh))
            lookup[(int(lat_idx), int(lon_idx))] = effective
        logging.info(f"Loaded {len(lookup)} per-station thresholds (abs_threshold={abs_thresh} m/s)")
        return lookup

    #### Target prep
    def time_to_data_grid(self, target_df):
        """перевести реальные временные метки станционных наблюдений в индексы ближайших узлов временной сетки модели"""
        start_time = time.process_time()   
        dates = target_df["time"].to_numpy()
        y = target_df["y"].to_numpy()
        values = round_to_closest_indices(dates, self.time_coords) # time into inds
        target_df = target_df.with_columns(
                            polars.Series(name="time", values=values),
                            polars.Series(name="y", values=y),
                            )
        logging.info(f"Time align took {time.process_time() - start_time} seconds")
        return target_df


    def stations_to_data_grid(self, stations_df: polars.DataFrame) -> polars.DataFrame:
        """ maps stations to the data grid pixels """
        start_time = time.process_time()   
        lat = stations_df["lat"].to_numpy() 
        lon = stations_df["lon"].to_numpy()
        lat_vector = round_to_closest_indices(lat, self.lat_coords)
        lon_vector = round_to_closest_indices(lon, self.lon_coords)
        stations_df = stations_df.with_columns(
                            [
                             polars.Series(name="lat", values=lat_vector),
                             polars.Series(name="lon", values=lon_vector)
                            ])
        logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")
        return stations_df
    

    def prepare_target_df(self):
        target_df = polars.read_parquet(os.path.join(self.cfg.train.data_dir, self.cfg.train.target_data_file)) # data/cmip5_world/target.parquet - станционные данные
        logging.info(f"Records before preparation {len(target_df)}")
        start_date = pd.to_datetime(self.cfg.train.start_time)
        end_date = pd.to_datetime(self.cfg.train.end_time)
        logging.info(f"Target time bounds before filter {target_df['time'].min()}, {target_df['time'].max()}")
        target_df = target_df.filter((polars.col('time') >= start_date) & (polars.col('time') < end_date)) # Может <= ??????
        logging.info(f"Target time bounds after filter {target_df['time'].min()}, {target_df['time'].max()}")
        logging.info(f"Data time bounds {self.time_coords.min()}, {self.time_coords.max()}") # self.time_coords = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        logging.info(f"Stations before aggregation: {target_df.n_unique(subset=['lat', 'lon'])}")
        target_df = self.time_to_data_grid(target_df)
        target_df = self.stations_to_data_grid(target_df)

        target_df = (target_df
                    .lazy()        
                    .sort("time")
                    .group_by(["lat", "lon", "time"])
                    .agg(
                        [
                         polars.col('y').quantile(0.65).alias("y"), # ?
                        ])
                    .collect())
        
        target_df = (target_df
                    .lazy()        
                    .sort("time")
                    .group_by(["lat", "lon"])
                    .agg(
                        [
                         polars.col("time"),
                         polars.col('y'),
                        ])
                    .collect())
        self.target_df = target_df.drop_nulls()
        logging.info(f"Stations after aggregation: {len(target_df)}")


    def target_df_to_array(self):
        """Берёт агрегированные станционные наблюдения, переводит в массив "индексированных"
        обучающих примеров, корректирует индексы под кроп/сдвиг и делит по времени:

            train: time <  start_of_val
            val:   start_of_val <= time < start_of_test
            test:  time >= start_of_test

        Если start_of_val не задан, val=test (старое поведение для обратной совместимости —
        НЕ рекомендуется, утечка).
        """
        start_time = time.process_time()
        test_split_date = datetime.strptime(self.cfg.train.start_of_test, '%Y-%m-%d').date()
        test_split_index = int(self.time_coords.searchsorted(test_split_date))
        val_split_str = self.cfg.train.get('start_of_val', None)
        if val_split_str is not None:
            val_split_date = datetime.strptime(val_split_str, '%Y-%m-%d').date()
            val_split_index = int(self.time_coords.searchsorted(val_split_date))
            if val_split_index >= test_split_index:
                raise ValueError(
                    f"start_of_val ({val_split_str}) must be strictly before start_of_test "
                    f"({self.cfg.train.start_of_test})."
                )
        else:
            val_split_index = None
            logging.warning(
                "start_of_val not set — validation will reuse the test set. "
                "This causes early-stopping to peek at test data; set start_of_val "
                "(e.g. '2021-01-01') to fix."
            )

        targets_list = []
        drop_dict = {}
        total = 0
        for target_df_row in self.target_df.rows():
            lat, lon, dates, y = target_df_row
            y = np.array(y)
            dates = np.array(dates)
            res = self.stations_filter(lat, lon, dates, y, self.cfg.target_type)
            if res != True:
                if res not in drop_dict:
                    drop_dict[res] = 1
                else:
                    drop_dict[res] += 1
                continue
            # Look up per-station effective threshold (before shift is applied to lat/lon)
            eff_thresh = self.station_threshold_lookup.get(
                (int(lat), int(lon)),
                float(self.cfg.train.get('abs_wind_threshold', self.cfg.train.target_threshold))
            )
            targets_list.append(self.pixel_aggregation(lat, lon, dates, y, eff_thresh))
            total += 1
        logging.info(f"Pixel loop took {time.process_time() - start_time} seconds, droped {drop_dict}")
        logging.info(f"Stations finally: {total}")
        target_array = np.concatenate(targets_list, axis=1)
        del targets_list
        target_array = target_array[:, ::self.cfg.train.time_freq]

        if self.cfg.train.spatial_crop:
            target_array[0, :] += self.shift[0] - self.lat_min_idx #lat
            target_array[1, :] += self.shift[1] - self.lon_min_idx #lon
        else:
            target_array[0, :] += self.shift[0] #lat
            target_array[1, :] += self.shift[1] #lon

        # Build splits — either standard temporal or geo-OOD (station hold-out).
        time_idx = target_array[2, :]
        n_time = len(self.time_coords)

        geo_ood = self.cfg.train.get('geo_ood_split', False)
        if geo_ood:
            # Geo-OOD: hold out a random fraction of stations entirely for test.
            # Station identity = (padded lat_idx, padded lon_idx) — unique per CMIP6 cell.
            geo_seed  = int(self.cfg.train.get('geo_ood_seed', 42))
            geo_ratio = float(self.cfg.train.get('geo_ood_test_ratio', 0.2))
            st_keys   = target_array[0].astype(np.int64) * 1000 + target_array[1].astype(np.int64)
            unique_st = np.unique(st_keys)
            n_test_st = max(1, int(len(unique_st) * geo_ratio))
            rng       = np.random.default_rng(geo_seed)
            test_st   = set(rng.choice(unique_st, size=n_test_st, replace=False).tolist())
            is_test_st = np.isin(st_keys, list(test_st))

            if val_split_index is not None:
                train_mask = (~is_test_st) & (time_idx < val_split_index)
                val_mask   = (~is_test_st) & (time_idx >= val_split_index) & (time_idx < test_split_index)
            else:
                train_mask = (~is_test_st) & (time_idx < test_split_index)
                val_mask   = (~is_test_st) & (time_idx >= test_split_index) & (time_idx < n_time)
            test_mask = is_test_st  # all time steps for held-out stations

            train_array            = target_array[:, train_mask]
            self.val_data_idxs     = target_array[:, val_mask]
            self.test_data_idxs    = target_array[:, test_mask]
            logging.info(
                f"Geo-OOD split: {n_test_st}/{len(unique_st)} stations held out for test "
                f"(seed={geo_seed}, ratio={geo_ratio:.0%})"
            )
        elif val_split_index is not None:
            # Standard temporal split with separate val period.
            train_mask = time_idx < val_split_index
            val_mask   = (time_idx >= val_split_index) & (time_idx < test_split_index)
            test_mask  = (time_idx >= test_split_index) & (time_idx < n_time)
            train_array            = target_array[:, train_mask]
            self.val_data_idxs     = target_array[:, val_mask]
            self.test_data_idxs    = target_array[:, test_mask]
        else:
            # Backward-compatible fallback (val == test).
            train_array = target_array[:, time_idx < test_split_index]
            self.test_data_idxs = target_array[:, (time_idx >= test_split_index) & (time_idx < n_time)]
            self.val_data_idxs  = self.test_data_idxs

        # Sanity: assert there's no overlap on (lat, lon, time) between splits.
        def _keys(arr):
            return arr[0].astype(np.int64) * (n_time * 1_000_000) + arr[1].astype(np.int64) * n_time + arr[2].astype(np.int64)
        train_keys = set(_keys(train_array).tolist())
        val_keys   = set(_keys(self.val_data_idxs).tolist())
        test_keys  = set(_keys(self.test_data_idxs).tolist())
        # Geo-OOD: train/val share no stations with test; time may overlap → only check (lat,lon,time).
        assert not (train_keys & val_keys),  "train and val overlap on (lat,lon,time)"
        assert not (train_keys & test_keys), "train and test overlap on (lat,lon,time)"
        assert not (val_keys   & test_keys), "val and test overlap on (lat,lon,time)"

        # Store full train array for per-epoch resampling
        self.neg_ratio = float(self.cfg.train.get('neg_subsample_ratio', None) or 0.0)
        if self.neg_ratio > 0:
            y_vals = train_array[7, :]
            thresh_vals = train_array[8, :] if train_array.shape[0] > 8 else np.full(y_vals.shape, self.cfg.train.target_threshold)
            self._pos_idxs = np.where(y_vals >= thresh_vals)[0]
            self._neg_idxs = np.where(y_vals < thresh_vals)[0]
            self._full_train_array = train_array
            self.train_data_idxs = self._subsample_negatives(epoch=0)
        else:
            self.train_data_idxs = train_array
        logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared val   {self.val_data_idxs.shape[1]}')
        logging.info(f'Records prepared test  {self.test_data_idxs.shape[1]}')

        print(f"Форма таргетов: {self.train_data_idxs[7, :].shape}. Примеры сырых таргетов: {self.train_data_idxs[7, :][:10].round(2)}")
        gc.collect()

    def _subsample_negatives(self, epoch: int):
        n_neg_keep = int(len(self._pos_idxs) * self.neg_ratio)
        rng = np.random.default_rng(epoch)
        neg_keep = rng.choice(self._neg_idxs, size=min(n_neg_keep, len(self._neg_idxs)), replace=False)
        keep = np.sort(np.concatenate([self._pos_idxs, neg_keep]))
        return self._full_train_array[:, keep]

    def resample_for_epoch(self, epoch: int):
        if self.neg_ratio > 0:
            self.train_data_idxs = self._subsample_negatives(epoch)
            logging.info(f"Epoch {epoch}: resampled train to {self.train_data_idxs.shape[1]} samples")

    def stations_filter(self, lat, lon, dates, y, target_type):
        if len(y) < max(self.cfg.train.time_agg_window, self.cfg.time_window):
            return "too short"
        if self.cfg.train.spatial_crop: 
            if lat < self.lat_min_idx or lat > self.lat_max_idx or lon < self.lon_min_idx or lon > self.lon_max_idx:
                # print(f"drop {lat, lon}")
                return "out of train area"

        if target_type == 'temp_c':
            if np.count_nonzero(y < -35)/y.size > 0.9:
                return "low temp"
            if np.count_nonzero(y > 40)/y.size > 0.5:
                return "high temp" 
        elif target_type == 'wind_ms':
            if np.count_nonzero(y < 2)/y.size > 0.9:
                return "low speed"
            if np.count_nonzero(y > 16)/y.size > 0.5:
                return "high speed"
        else: 
            raise NotImplementedError
        return True


    def pixel_aggregation(self, lat, lon, dates, y, effective_threshold=None):
        if self.cfg.train.get('synthetic_labels', False):
            lat_pad = lat + self.shift[0]
            lon_pad = lon + self.shift[1]
            y = self.sfcwindmax_for_labels[dates, lat_pad, lon_pad].numpy().astype(np.float32)
        # aggregate target with given time_agg_window
        time_positions_m = np.array([d.astype(object).month for d in self.time_coords[dates]])
        time_positions_days =  np.array([d.astype(object).day for d in self.time_coords[dates]])
        time_positions = (time_positions_m * 30.5 + time_positions_days)/365
        time_positions_m = time_positions_m/12
        assert len(time_positions) == len(dates)
        # y_max: max wind speed over the 28-day sliding window
        # "was there at least one stormy day in this period?" → clean, interpretable for a paper
        y_max = np.max(sliding_window_view(y, window_shape=self.cfg.train.time_agg_window), axis=1)
        i = 1 if self.cfg.train.time_agg_window % 2 == 0 else 0
        if self.cfg.time_window > self.cfg.train.time_agg_window:
            # clip dates according to time_window
            dates = dates[self.cfg.time_window//2: len(dates)-self.cfg.time_window//2 + i]
            time_positions = time_positions[self.cfg.time_window//2: len(time_positions)-self.cfg.time_window//2 + i]
            time_positions_m = time_positions_m[self.cfg.time_window//2: len(time_positions_m)-self.cfg.time_window//2 + i]
            y_max = y_max[self.cfg.time_window-self.cfg.train.time_agg_window:
                          len(y_max) + self.cfg.train.time_agg_window - self.cfg.time_window - 1]
        else:
            dates = dates[self.cfg.train.time_agg_window//2: len(dates)-self.cfg.train.time_agg_window//2 + i]
            time_positions = time_positions[self.cfg.train.time_agg_window//2: len(time_positions)-self.cfg.train.time_agg_window//2 + i]
            time_positions_m = time_positions_m[self.cfg.train.time_agg_window//2: len(time_positions_m)-self.cfg.train.time_agg_window//2 + i]
            # y_max not changed

        assert len(time_positions) == len(dates)
        mask = dates > self.cfg.time_window//2+1
        dates = dates[mask]
        time_positions = time_positions[mask]
        time_positions_m = time_positions_m[mask]
        y_max = y_max[mask]

        lat_position = self.lat_coords[lat]/90
        lon_position = self.lon_coords[lon]/180
        # Rows 0-6: indices and positional encoding; Row 7: max wind over window; Row 8: effective threshold
        target_array = np.stack([np.full(len(dates), lat),
                                 np.full(len(dates), lon),
                                 dates,
                                 time_positions,
                                 time_positions_m,
                                 np.full(len(dates), lat_position),
                                 np.full(len(dates), lon_position),
                                 y_max,
                                 ])
        # Row 8: per-station effective threshold (constant for all time steps of this station)
        if effective_threshold is None:
            effective_threshold = float(self.cfg.train.target_threshold)
        target_array = np.concatenate(
            (target_array, np.full((1, target_array.shape[1]), effective_threshold, dtype=np.float32)),
            axis=0
        )
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
        val_size = self.val_data_idxs.shape[1] if hasattr(self, 'val_data_idxs') else 0
        logging.info(f"Train size: {self.train_data_idxs.shape[1]}, "
                     f"val size: {val_size}, "
                     f"test size: {self.test_data_idxs.shape[1]}")
        logging.info(f"Target min: {self.train_data_idxs[7, :].min()}, target max: {self.train_data_idxs[7, :].max()}")
        logging.info(f"Target mean: {self.train_data_idxs[7, :].mean()}, target std: {self.train_data_idxs[7, :].std()}")
        balance_msg = (f"Balance train: {self.get_class_balance(self.train_data_idxs[7, :])}, "
                       f"balance test: {self.get_class_balance(self.test_data_idxs[7, :])}")
        if hasattr(self, 'val_data_idxs') and self.val_data_idxs is not self.test_data_idxs:
            balance_msg = (f"Balance train: {self.get_class_balance(self.train_data_idxs[7, :])}, "
                           f"balance val: {self.get_class_balance(self.val_data_idxs[7, :])}, "
                           f"balance test: {self.get_class_balance(self.test_data_idxs[7, :])}")
        logging.info(balance_msg)
        for i, var in enumerate(self.cfg.train.variables):
            logging.info(f"{var} mean: {self.dataset_torch[i].mean()}, std: {self.dataset_torch[i].std()}")

    def get_class_balance(self, target_array):
        positive = np.sum(target_array >= self.cfg.train.target_threshold)
        all = target_array.shape[0]
        return positive/all
            
    
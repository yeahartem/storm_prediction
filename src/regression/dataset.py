import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset
from src.data_assemble.assemble_data import make_blocks_numpy
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars
import logging



def prepare_data(cfg):

    logging.info('tmp file not found, processing')
    start_time = time.process_time()
    time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
    lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
    lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
    var_data = np.empty((len(cfg.variables), len(time_coords), len(lat_coords), len(lon_coords)), dtype=np.float32)

    for i, var in enumerate(cfg.variables):
        var_data[i] = np.load(os.path.join(cfg.data_dir, var + f'_{16}.npy'))

    var_data = np.moveaxis(var_data, 0, 1)
    dataset_as_blocks = make_blocks_numpy(var_data, cfg.half_side_size, time_stack_size=cfg.time_window, time_freq=cfg.time_freq)  
    logging.info(f"Time to load and prep climate data {time.process_time() - start_time} seconds")

    def max_window(values):
        if len(values)<cfg.time_window:
            return None
        else:
            return np.max(sliding_window_view(np.array(values), window_shape = cfg.time_window), axis = 1)
        
    def align_time(values):
        if len(values)<cfg.time_window:
            return None
        else:
            values = time_coords.searchsorted(values) # time into inds
            return np.array(values)[cfg.time_window//2:len(values) - cfg.time_window//2]
    
    def align_coords(values):
        lat, lon = values - cfg.half_side_size
        if (lat < 0) or (lon < 0):
            return None
        elif (lat > (len(lat_coords) - 2*cfg.half_side_size)) or (lon > (len(lon_coords) - 2*cfg.half_side_size)):
            return None
        else:
            return np.array((lat, lon))
        
    start_time = time.process_time()  
    target_df = polars.read_parquet(cfg.path_to_prepared_target_data)
    logging.info(f"Records before preparation {len(target_df)}")

    target_df = (
        target_df
        .lazy()        
        .sort("time")
        .groupby(["station_name"])
        .agg(
            [polars.col('time').apply(align_time), polars.col('y').apply(max_window)]
        )
        .collect()
    )

    target_df = target_df.with_columns(polars.col('station_name').apply(align_coords).keep_name())
    logging.info(f"Stations before droppping: {len(target_df)}")
    logging.info(f"Time to prepare target {time.process_time() - start_time} seconds")
    split_date = datetime.strptime(cfg.start_of_test, '%Y-%m-%d').date()
    split_index = time_coords.searchsorted(split_date)

    train_data_idxs = []
    test_data_idxs = []

    for coords, dates, y in target_df.rows():
        if (coords is not None) and (dates is not None) and (y is not None):
            if not (any(np.isnan(np.array(coords), casting='unsafe')) and any(np.isnan(np.array(dates), casting='unsafe')) and any(np.isnan(np.array(y), casting='unsafe'))):
                clipped_dates = dates[dates < (dataset_as_blocks.shape[2]-1)]
                dates_train = clipped_dates[clipped_dates < split_index]
                dates_test = clipped_dates[clipped_dates >= split_index]

                y_train = y[:len(dates_train)]
                y_test = y[len(dates_train):len(clipped_dates)]

                arr_train = np.stack([np.full(len(dates_train),coords[0], dtype=np.int16), np.full(len(dates_train), coords[1], dtype=np.int16), dates_train, y_train])
                arr_test = np.stack([np.full(len(dates_test),coords[0], dtype=np.int16), np.full(len(dates_test), coords[1], dtype=np.int16), dates_test, y_test])

                train_data_idxs.append(arr_train)
                test_data_idxs.append(arr_test)
    
    train_data_idxs = np.concatenate(train_data_idxs, axis=1)
    test_data_idxs = np.concatenate(test_data_idxs, axis=1)
    logging.info(f'Records prepared train {train_data_idxs.shape[1]}')
    logging.info(f'Records prepared test {test_data_idxs.shape[1]}')

    return dataset_as_blocks, train_data_idxs, test_data_idxs



class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = int(os.environ.get("NODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables
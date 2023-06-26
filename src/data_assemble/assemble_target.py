import sys,os
sys.path.append(os.getcwd())
import pandas as pd
from functools import partial
import numpy as np
import logging
import xarray as xr
import time
from omegaconf import DictConfig, ListConfig
from src.data_assemble.prepare_target import get_stations_RU
import gc
import polars as pl


def preprocess_coordinates():
    pass

    

def closest_pixel_for_station(station_row, grid_lat, grid_lon, precise=False):

    """Finds the closest pixel in the dataset for the station
    Returns:
    tuple: x, y lat and lon of the closest pixel on data grid
    """
    coord = [station_row["lat"], station_row["lon"]]        
    theta_offset = 5
    fi_offset = 5
    nearest_lat_idx = (np.abs(np.asarray(grid_lat) - coord[0])).argmin() # find nearest
    nearest_lon_idx = (np.abs(np.asarray(grid_lat) - coord[1])).argmin()

    if precise:
        closest = great_circle((grid_lat[nearest_lat_idx], grid_lon[nearest_lon_idx]), coord).kilometers
        for i, theta in enumerate(grid_lon[nearest_lon_idx - 5:nearest_lon_idx + 5]): # find the closest pixel in 10x10 grid by circle distance            
            for j, fi in enumerate(grid_lat[nearest_lat_idx - 5:nearest_lat_idx + 5]):
                r = great_circle((fi, theta), coord).kilometers
                if r < closest:
                    closest = r
                    theta_offset = i
                    fi_offset = j

    closest_x_idx = theta_offset - 5 + nearest_lon_idx
    closest_y_idx = fi_offset - 5 + nearest_lat_idx

    return grid_lon[closest_x_idx], grid_lat[closest_y_idx]    



def round_to_closest_values(arr, values):

    values = np.array(values)
    indices = np.searchsorted(values, arr)
    indices = np.clip(indices, 1, len(values) - 1)
    left_values = values[indices - 1]
    right_values = values[indices]
    closest_values = np.where(np.abs(arr - left_values) <= np.abs(arr - right_values), left_values, right_values)
    return closest_values


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


def stations_to_data_grid(
        dataset_xarray: xr.DataArray,
        stations_df: pl.DataFrame) -> pl.DataFrame:
    """
    maps stations to the data grid pixels
    """
    grid_lon = dataset_xarray.lon.data
    grid_lat = dataset_xarray.lat.data
    start_time = time.process_time()   
    lat_vector = round_to_closest_indices(stations_df["lat"].to_numpy(), grid_lat)
    lon_vector = round_to_closest_indices(stations_df["lon"].to_numpy(), grid_lon)
    stations_df = stations_df.with_columns(
                        pl.Series(name="lat", values=lat_vector),
                        pl.Series(name="lon", values=lon_vector)
                        )
    
    logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")

    return stations_df   

def filter_lat_lon(stations_df, cfg):
    lat_min, lat_max, lon_min, lon_max = cfg.train_coords.lat_min, cfg.train_coords.lat_max, cfg.train_coords.lon_min, cfg.train_coords.lon_max
    return stations_df.filter(pl.any((pl.col('lat') >= lat_min) & (pl.col('lat') <= lat_max) & (pl.col('lon') <= lon_max) & (pl.col('lon') >= lon_min)))

def pre_prepare_target_RU(cfg: DictConfig, dataset_xarray: xr.DataArray):
        
    start_time = time.process_time()
    df = pl.read_parquet(cfg.path_to_weather_stations_data.replace(".parquet", "_cleaned.parquet"), use_pyarrow=True)
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")

    start = cfg.time_limits[0]
    end = cfg.time_limits[1]

    df = df.filter(pl.any(pl.col('time') >= pd.to_datetime(start)))
    df = df.filter(pl.any(pl.col('time') <= pd.to_datetime(end)))
    
    if len(cfg.target_column)>1:
        target_cols = [df[col] for col in cfg.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.target_column)==1:
        df = df.rename({cfg.target_column[0]: 'y'})    
    else:
        raise ValueError
    
    df = df.select(pl.col(["time", "station_name", "y" ]))
    gc.collect()
    
    stations_df_ru = pl.from_pandas(get_stations_RU(cfg))
    stations_df_ru = filter_lat_lon(stations_df_ru, cfg)      
    stations_df_ru = stations_to_data_grid(dataset_xarray=dataset_xarray,
                                           stations_df=stations_df_ru)
          
    df = df.select([pl.all().exclude("station_name"), pl.col("station_name").cast(str).keep_name()])   
    print(stations_df_ru)

    df = df.join(stations_df_ru, on='station_name', how='left')
    df = df.select(pl.col(["time", "y", "lat", "lon"]))
    print(df)
    gc.collect()
    # df = df.select([pl.all().exclude("station_name"), pl.col("station_name").cast(pl.Categorical).keep_name()])

    df.write_parquet(os.path.join(cfg.data_dir, cfg.prepared_target_data_name + '.pp1'))

    

def pre_prepare_target_WORLD(cfg: DictConfig, dataset_xarray: xr.DataArray):

    start_time = time.process_time()
    df = pl.read_parquet(cfg.path_to_world_weather_stations_data.replace(".parquet", "_cleaned.parquet"), use_pyarrow=True)
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")

    start = cfg.time_limits[0]
    end = cfg.time_limits[1]
    df = df.filter(pl.any(pl.col('time') >= pd.to_datetime(start)))
    df = df.filter(pl.any(pl.col('time') <= pd.to_datetime(end)))
    df = filter_lat_lon(df, cfg)

    if len(cfg.target_column)>1:
        target_cols = [df[col] for col in cfg.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.target_column)==1:
        df = df.rename({cfg.target_column[0]: 'y'})    
    else:
        raise ValueError
    
    df = df.select(pl.col(["time", "station_name", "y", "lat", "lon"]))
    gc.collect()         

    df = stations_to_data_grid(dataset_xarray=dataset_xarray, stations_df=df)
    df = df.select([pl.all().exclude("station_name"), pl.col("station_name").cast(str).keep_name()])
    # df = df.select([pl.all().exclude("station_name")])
    df.write_parquet(os.path.join(cfg.data_dir, cfg.prepared_target_data_name + '.pp2'))



def make_target(cfg: DictConfig, dataset_xarray: xr.DataArray):    

    pre_prepare_target_RU(cfg, dataset_xarray)
    pre_prepare_target_WORLD(cfg, dataset_xarray)

    df_ru = pl.read_parquet(os.path.join(cfg.data_dir, cfg.prepared_target_data_name + '.pp1'), use_pyarrow=True)
    logging.info(f'RU len: {len(df_ru)}')
    df_world = pl.read_parquet(os.path.join(cfg.data_dir, cfg.prepared_target_data_name + '.pp2'), use_pyarrow=True)
    logging.info(f'WORLD len: {len(df_world)}')

    start_time = time.process_time()
    target_df = pl.concat([df_ru, df_world], how='diagonal')


    target_df = target_df.with_columns(
        [
        pl.col("lat").cast(pl.Int16).alias('lat'),
        pl.col("lon").cast(pl.Int16).alias('lon'),
        pl.concat_list(pl.col('lat'), pl.col('lon')).alias('station_name')
        ]
    )    
    logging.info(f"Concat took {time.process_time() - start_time} seconds")
    logging.info(f'TOTAL len: {len(target_df)}')
    target_df = target_df.drop_nulls()
    target_df.write_parquet(os.path.join(cfg.data_dir, cfg.prepared_target_data_name))
import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import logging
import time
import polars as pl

def cleanup_ms_name(name: str):        
        name = name.replace('"', '')
        name = name.replace(',', '')
        name = name.casefold()
        return name

def get_stations_RU(cfg) -> pd.DataFrame:

    all_stations_data = cfg.path_to_weather_station_list
    rectangle_coords = list(cfg.train_coords.values())                                   
    min_lat=rectangle_coords[0]
    max_lat=rectangle_coords[1]                                    
    min_lon=rectangle_coords[2]
    max_lon=rectangle_coords[3]
    max_height = cfg.max_height
    min_height = cfg.min_height
    all_stations = pd.read_json(all_stations_data)
    all_stations = all_stations.rename(columns={
                            "Широта": "lat",
                            "Долгота": "lon",
                            "Наименование станции": "station_name",
                            "Высота метеопл.": "height"
                            })    
    
    all_stations["station_name"] = all_stations["station_name"].apply(cleanup_ms_name)
    all_stations["lat"] = all_stations["lat"].astype(np.float32)
    all_stations["lon"] = all_stations["lon"].astype(np.float32)
    all_stations["height"] = all_stations["height"].astype(np.float32)

    result_stations = all_stations[["station_name", "lat", "lon", "height"]]

    if cfg.spatial_crop:
        if max_lat:
            result_stations = result_stations[result_stations['lat'] < max_lat]
        if min_lat:
            result_stations = result_stations[result_stations['lat'] > min_lat]
        if max_lon:
            result_stations = result_stations[result_stations['lon'] < max_lon]
        if min_lon:
            result_stations = result_stations[result_stations['lon'] > min_lon]
        if max_height:
            result_stations = result_stations[result_stations['height'] < max_height]
        if min_height:
            result_stations = result_stations[result_stations['height'] > min_height]
    
    return result_stations



def clean_weather_data_RU(path_to_weather_stations: str) -> pd.DataFrame:
    """ 
    To load weather stations data from Russia 
        Features: ['Максимальная скорость ветра', 'Средняя скорость ветра', 'Направление ветра', 
                   'Температура воздуха по сухому терм-ру', 'Атмосферное давление на уровне станции', 
                   'Атмосферное давление на уровне моря', 'Сумма осадков', 'Температура поверхности почвы',
                   'Парциальное давление водяного пара', 'Относительная влажность воздуха', 'Температура точки росы'] 
    """  

    

    columns = ["Название метеостанции",
               "Максимальная скорость",
               'Средняя скорость ветра',
               'Температура воздуха по сухому терм-ру',
               'Температура точки росы',
               'Атмосферное давление на уровне станции',
               'Атмосферное давление на уровне моря',
               "Дата"] # Add more columns if needed
    
    start_time = time.process_time()
    df = pl.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")
    
    q = (df
        .lazy()
        .select(
            [
                pl.col("Название метеостанции").apply(cleanup_ms_name).cast(pl.Categorical).alias("station_name"),
                pl.col("Максимальная скорость").round().cast(pl.UInt8).alias("max_speed"),
                pl.col("Средняя скорость ветра").round().cast(pl.UInt8).alias("avg_speed"),
                pl.col("Дата").str.strptime(pl.Date, fmt="%Y-%m-%d", strict=False).alias("time"),              
                pl.col("Температура воздуха по сухому терм-ру").round().cast(pl.Int16).alias("avg_temp"),
                pl.col("Температура точки росы").round().cast(pl.Int16).alias("dew_point_temp"),
                pl.col("Атмосферное давление на уровне станции").round().cast(pl.Int16).alias("station_level_pressure"),
                pl.col("Атмосферное давление на уровне моря").round().cast(pl.Int16).alias("sea_level_pressure"),
            ]
               )
        )
    
    q = q.collect()
    q.write_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))



def clean_weather_data_WORLD(path_to_weather_stations: str) -> pd.DataFrame:
    """ To load weather stations data from all world 
        Features: ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'STP', 'SLP',
       'PRCP', 'DEWP', 'LATITUDE', 'LONGITUDE', 'ELEVATION'] 
    """  

    columns = ["STATION", "LATITUDE",  "LONGITUDE", "ELEVATION", "DATE", 'MXWDSP', 'WDSP', 'TEMP', 'DEWP', 'SLP', 'STP'] 
    start_time = time.process_time()
    df = pl.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")
    
    q = (df
        .lazy()
        .select(
            [
                pl.col("STATION").cast(pl.Categorical).alias("station_name"),
                pl.col("MXWDSP").round().cast(pl.UInt8).alias("max_speed"),
                pl.col("WDSP").round().cast(pl.UInt8).alias("avg_speed"),
                pl.col("DATE").cast(pl.Date).alias("time"),                
                pl.col("TEMP").round().cast(pl.Int16).alias("avg_temp"),
                pl.col("DEWP").round().cast(pl.Int16).alias("dew_point_temp"),
                pl.col("STP").round().cast(pl.Int16).alias("station_level_pressure"),
                pl.col("SLP").round().cast(pl.Int16).alias("sea_level_pressure"),
                pl.col("LATITUDE").round().cast(pl.Float32).alias("lat"),
                pl.col("LONGITUDE").round().cast(pl.Float32).alias("lon"),
                pl.col("ELEVATION").round().cast(pl.Float32).alias("height"),
            ]
               )
        )
    
    q = q.collect()
    q.write_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))



def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
# stations_df_world = pd.DataFrame([{'lon': df[df['station_name'] == name].iloc[0]['lon'],
#                               'lat': df[df['station_name'] == name].iloc[0]['lat'],
#                               'height': df[df['station_name'] == name].iloc[0]['height'],
#                               'station_name': name}
#                                for name in df['station_name'].unique()])
# stations_df_world = stations_to_data_grid(dataset_xarray=dataset_xarray,
#                                       stations_df=stations_df_world)    
# df = df.merge(stations_df_world, on='station_name', how='left')
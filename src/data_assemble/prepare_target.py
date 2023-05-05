import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import logging
import time


def get_stations_RU(cfg) -> pd.DataFrame:

    all_stations_data = cfg.path_to_weather_station_list

    stations_allowed_path=cfg.path_to_allowed_stations 
    rectangle_coords = list(cfg.train_coords.values())                                   
    min_lat=rectangle_coords[0]
    max_lat=rectangle_coords[1]                                    
    min_lon=rectangle_coords[2]
    max_lon=rectangle_coords[3]
    max_height = cfg.max_height
    min_height = cfg.min_height

    all_stations = pd.read_json(all_stations_data)
    with open(stations_allowed_path) as f:
        stations_allowed = f.read().split('\n')

    all_stations = all_stations.rename(columns={
                            "Широта": "lat",
                            "Долгота": "lon",
                            "Наименование станции": "station_name",
                            "Высота метеопл.": "height"
                            })    
    
    all_stations = all_stations[["station_name", "lat", "lon", "height"]]
    result_stations = all_stations[all_stations['station_name'].isin(stations_allowed)]
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
    result_stations['station_name'] = result_stations['station_name'].str.casefold()
    
    return result_stations



def clean_weather_data_RU(path_to_weather_stations: str) -> pd.DataFrame:
    """ 
    To load weather stations data from Russia 
        Features: ['Максимальная скорость ветра', 'Средняя скорость ветра', 'Направление ветра', 
                   'Температура воздуха по сухому терм-ру', 'Атмосферное давление на уровне станции', 
                   'Атмосферное давление на уровне моря', 'Сумма осадков', 'Температура поверхности почвы',
                   'Парциальное давление водяного пара', 'Относительная влажность воздуха', 'Температура точки росы'] 
    """  

    def cleanup_ms_name(name: str):
        name = name.replace('"', '')
        name = name.replace(',', '')
        return name

    columns = ["Название метеостанции",
               "Максимальная скорость",
               'Средняя скорость ветра',
               'Температура поверхности почвы',
               'Температура точки росы',
               'Парциальное давление водяного пара',
               'Атмосферное давление на уровне моря',
               "Дата"] # Add more columns if needed
    
    start_time = time.process_time()
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")

    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df['Название метеостанции'] = df['Название метеостанции'].str.casefold()
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df["Максимальная скорость"] = df[["Максимальная скорость"]].apply(pd.to_numeric, downcast="float")
    df['Дата'] = pd.DatetimeIndex(df['Дата'])
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")    
    df = df.rename(columns={"Дата": "time",
                            "Название метеостанции": "station_name",
                            "Максимальная скорость": "max_speed",
                            })        
    df.to_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))



def clean_weather_data_WORLD(path_to_weather_stations: str) -> pd.DataFrame: # TODO add target colums argument
    """ To load weather stations data from all world 
        Features: ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'STP', 'SLP',
       'PRCP', 'DEWP', 'LATITUDE', 'LONGITUDE', 'ELEVATION'] 
    """  

    columns = ["STATION", "MXWDSP", "LATITUDE",  "LONGITUDE", "ELEVATION", 'TEMP',  'DEWP', 'PRCP', "DATE"] 
    start_time = time.process_time()
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")

    df['STATION'] = df['STATION'].str.casefold()
    df["STATION"] = df["STATION"].astype("category")
    df['DATE'] = pd.DatetimeIndex(df['DATE'])
    df["DATE"] = pd.to_datetime((df["DATE"]), format="%Y-%m-%d")    

    df["MXWDSP"] = df["MXWDSP"].apply(pd.to_numeric, downcast="float")
    df = df.rename(columns={"DATE": "time",
                            "STATION": "station_name",
                            "MXWDSP": "max_speed",
                            "LATITUDE": "lat",
                            "LONGITUDE": "lon",
                            "ELEVATION": "height"
                            })    
    
    df.to_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))


# stations_df_world = pd.DataFrame([{'lon': df[df['station_name'] == name].iloc[0]['lon'],
#                               'lat': df[df['station_name'] == name].iloc[0]['lat'],
#                               'height': df[df['station_name'] == name].iloc[0]['height'],
#                               'station_name': name}
#                                for name in df['station_name'].unique()])
# stations_df_world = stations_to_data_grid(dataset_xarray=dataset_xarray,
#                                       stations_df=stations_df_world)    
# df = df.merge(stations_df_world, on='station_name', how='left')
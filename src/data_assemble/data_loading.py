import pandas as pd
import json
import os
import sys
import numpy as np


def processing(v):
  if type(v) == float or type(v) == int:
        return v
  elif v.strip() == '':
        return np.nan
  else:
        return float(v.strip())


def weatherstations_loader(
    path_to_data: str = '../../../stash/kurdyukova/data_meteo_full/sroks',
    path_to_station_dict: str = '../../../stash/kurdyukova/data_meteo_full/station_dict_full.json',
    to_save: bool = False,
    path_for_saving: str = '../../../stash/kurdyukova/data_meteo_full/data_meteo_full.csv',
    path_to_column_dict: str = '../../../stash/kurdyukova/data_meteo_full/col_num.json',
    features: list = ['Максимальная скорость ветра', 'Средняя скорость ветра', 'Направление ветра', 
                        'Температура воздуха по сухому терм-ру', 'Атмосферное давление на уровне станции', 
                        'Атмосферное давление на уровне моря', 'Сумма осадков', 'Температура поверхности почвы',
                        'Парциальное давление водяного пара', 'Относительная влажность воздуха', 'Температура точки росы'] 
):
    with open(path_to_station_dict, "r") as my_file:
        station_id_json = my_file.read()
    station_id_dict = json.loads(station_id_json)

    with open(path_to_column_dict, "r") as my_file:
        col_name_json = my_file.read()
    col_name_dict = json.loads(col_name_json)

    stations_dict = dict({})

    dir_list = os.listdir(path_to_data)
    for dir in dir_list:
        if dir[:2] == 'wr':
            file_list = os.listdir(path_to_data + '/' + dir)
            for file in file_list:
                if len(file[:-4]) == 5 and file[-3:] == 'txt':
                    Station = pd.read_table(path_to_data + '/' + dir + '/' + file, sep=';', header=None)
                    print(path_to_data + '/' + dir + '/' + file)
                    Station_date = pd.DataFrame({'year': Station.iloc[:,col_name_dict['Год по Гринвичу']],
                                                'month': Station.iloc[:,col_name_dict['Месяц по Гринвичу']],
                                                'day': Station.iloc[:,col_name_dict['День по Гринвичу']]})
                    Data_Station = pd.DataFrame({'Дата': pd.to_datetime(Station_date)})
                    Data_Station = Data_Station.join(pd.DataFrame({'Название метеостанции': [station_id_dict[file[:-3]+'0'] for i in range(len(Station))]}))
                    for feature in features:
                        Data_Station = Data_Station.join(pd.DataFrame({feature: Station.iloc[:,col_name_dict[feature]]}))
                    for col in Data_Station.columns:
                        if col not in ['Дата', 'Название метеостанции']:     
                            Data_Station[col] = Data_Station[col].apply(processing)
                    stations_dict[station_id_dict[file[:-3]+'0']] = Data_Station
                    
    data_meteo_full = pd.concat(stations_dict.values())

    if to_save:
        data_meteo_full.to_csv(path_for_saving, index=False)

    return data_meteo_full
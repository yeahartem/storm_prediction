
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from src.utils import data_processing as dp

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import random
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")

def make_ds(start_date, end_date, station_name, station_list, ds_for_y, cmip, include_day_ohe=False, speed_th=20, pix=[-1, -1]):
    '''
    Makes dataset consisting of X from make_model_dataset(weatherstation_list.csv) and y from data_meteo_kk.csv.
    
    station_list - weatherstation_list.csv,
    ds_for_y - data_meteo_kk.csv.
    '''
    a = make_model_dataset(station_name = station_name, start_date = start_date, wind_cmip=cmip, end_date = end_date, station_list = station_list, pix=pix)
    a.insert(loc=0, column='day', value=a.index)
    a['day'] = a['day'].dt.dayofyear
    
    if include_day_ohe:
        enc = OneHotEncoder()
        enc.fit(np.array(a.day).reshape(-1, 1))
        a.drop('day', axis=1, inplace=True)
        lis = [[enc.transform(np.atleast_2d(a.iloc[i].name.dayofyear)).toarray()] for i in range(a.shape[0])]
        ohe_day = pd.DataFrame(np.array(lis).reshape(np.array(lis).shape[0], -1))
        X = pd.concat([a, ohe_day.set_index(a.index)], axis=1)
    else:
        a.drop('day', axis=1, inplace=True)
        X = a
    

    # X = pd.concat([a, ohe_day.set_index(a.index)], axis=1)
    

    ds_for_y['Дата'] = pd.to_datetime((ds_for_y['Дата']), format="%Y/%m/%d")
    ds_for_y = ds_for_y.loc[(ds_for_y['Дата'] >= pd.to_datetime(start_date)) & (ds_for_y['Дата'] <= pd.to_datetime(end_date))]
    ds_for_y = ds_for_y.loc[ds_for_y['Название метеостанции'] == station_name]
    ds_for_y = ds_for_y.groupby(ds_for_y['Дата']).max()   # Как группируется?
    max_speed = ds_for_y[['Максимальная скорость', 'Средняя скорость ветра']].max(axis=1)
    y = np.array((max_speed >= speed_th).astype(int))

    return X, y


def dss(start_date, end_date, station_name, station_list, ds_for_y, delta_in_x, cmip, include_day_ohe=False, speed_th=20, pix=[-1, -1]):
    '''
    Includes several days into X. Makes dataset consisting of X from make_model_dataset(weatherstation_list.csv) and y from data_meteo_kk.csv.
    
    station_list - weatherstation_list.csv,
    ds_for_y - data_meteo_kk.csv,
    delta_in_x - number of days into X.
    '''
    a = make_model_dataset(station_name = station_name, start_date = start_date, end_date = end_date, wind_cmip=cmip, station_list = station_list, pix=pix)
    a.insert(loc=0, column='day', value=a.index)
    a['day'] = a['day'].dt.dayofyear

    if include_day_ohe:
        enc = OneHotEncoder()
        enc.fit(np.array(a.day).reshape(-1, 1))
        a.drop('day', axis=1, inplace=True)
        lis = [
            [
                np.concatenate(
                    (
                        np.array(a.iloc[i-delta_in_x : i]).reshape(1, -1),
                        enc.transform(
                            np.atleast_2d(a.iloc[i-1].name.dayofyear)
                        ).toarray(),
                    ),
                    axis=1,
                )
            ]
            for i in range(delta_in_x, a.shape[0])
        ]
    else:
        a.drop('day', axis=1, inplace=True)
        lis = [
            [
                np.concatenate(
                    (
                        np.array(a.iloc[i-delta_in_x : i]).reshape(1, -1),
                    ),
                    axis=1,
                )
            ]
            for i in range(delta_in_x, a.shape[0])
        ]
    
    X = np.array(lis).reshape(np.array(lis).shape[0], -1)

    ds_for_y['Дата'] = pd.to_datetime((ds_for_y['Дата']), format="%Y/%m/%d")
    ds_for_y = ds_for_y.loc[(ds_for_y['Дата'] >= pd.to_datetime(start_date)) & (ds_for_y['Дата'] <= pd.to_datetime(end_date))]
    ds_for_y = ds_for_y.loc[ds_for_y['Название метеостанции'] == station_name]
    ds_for_y = ds_for_y.groupby(ds_for_y['Дата']).max()   # Как группируется?
    max_speed = ds_for_y[['Максимальная скорость', 'Средняя скорость ветра']].max(axis=1)
    y = np.array((max_speed >= speed_th).astype(int))
    y = y[delta_in_x:a.shape[0]]

    return X, y
import numpy as np 
import pandas as pd
import datetime


def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def test_climatedata_to_patches():

    # create dummy data
    climate_vars = ['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6' ]
    var_1 = np.arange(0, 100, 1).reshape(10, 10) + 0.1
    var_2 = np.arange(0, 100, 1).reshape(10, 10) + 0.2
    var_3 = np.arange(0, 100, 1).reshape(10, 10) + 0.3
    var_4 = np.arange(0, 100, 1).reshape(10, 10) + 0.4
    var_5 = np.arange(0, 100, 1).reshape(10, 10) + 0.5
    var_6 = np.arange(0, 100, 1).reshape(10, 10) + 0.6

    time_slice = np.stack([var_1, var_2, var_3, var_4, var_5, var_6], axis=0)
    data = np.stack([time_slice + 0.001*(i+1) for i in range(60)], axis=0)

    start_date = str_to_date('2020-12-02')
    time_coords = [start_date + datetime.timedelta(days=x) for x in range(60)]  
    lat_coords = np.arange(0, 12, 1.25)
    lon_coords = np.arange(0, 11, 1.15)

    assert data.shape[0] == len(time_coords)
    assert data.shape[1] == len(climate_vars)
    assert data.shape[2] == len(lat_coords)
    assert data.shape[3] == len(lon_coords)

    print(f"climate data shape: {data.shape}")
    print(f"time coords shape: {len(time_coords)}")
    print(f"lat coords: {lat_coords}")
    print(f"lon coords: {lon_coords}")
    # create dummy target

    df_data = {"time": [0, 2],
               "y": [3, 7],
               "lat": [3, 7],
               "lon": [3, 7]
              }
    

if __name__ == "__main__":
    test_climatedata_to_patches()
from tkinter import Y
import sys
sys.path.append('..')
from src.data_utils.utils import check_leap_year

import matplotlib.pyplot as plt
import os, glob
import numpy as np
from tqdm import tqdm
import subprocess
import pandas as pd
from geopy.distance import great_circle
from sklearn import preprocessing
from scipy import interpolate
import xarray

# from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians

def get_xarrays(path_to_data: str, rectangle_coords: dict, target_res: dict, contains: dict = {"years": ['2006'], "bands": ['max']}, time_limits: dict = {'t_start': np.datetime64('2005-01-01'), 't_end': np.datetime64('2020-01-01')}) -> dict:
    '''
    Returns reduced to rectangle_coords xarrays.

    rectangle_coords - {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40},
    target_res - {'lon_res': 0.25, 'lat_res': 0.25},
    contains - things which should be included into the titles of .nc files: {"years": ['2016', '2026'], "bands": ['max', 'Wind_']}
    '''
    #unpack required geometry, target resolutions
    lat_min = rectangle_coords['lat_min']
    lat_max = rectangle_coords['lat_max']
    lon_max = rectangle_coords['lon_max']
    lon_min = rectangle_coords['lon_min']
    lat_res = target_res['lat_res']
    lon_res = target_res['lon_res']
    t_start = time_limits["t_start"]
    t_end = time_limits["t_end"]

    xarrays = {}
    for band in contains["bands"]:
        band_year = []
        for year in contains["years"]:
            f1_xarray = open_dataxarray(path_to_data, [year, band])
            f1_xarray = reduce_to_area(f1_xarray, lat_min, lat_max, lon_min, lon_max) 
            f1_xarray_refined = interp_timewise_xarray(f1_xarray, lon_res=lon_res, lat_res=lat_res, interp_method = 'linear', plot_example = False)
            band_year.append(f1_xarray_refined)
        band_year = xarray.concat(band_year, dim="time") # stack xarrays on time axis
        
        band_year = band_year.sel(time=slice(t_start, t_end))
        xarrays[band] = band_year
    return xarrays


def open_dataxarray(path_to_data: str, contains: list = ['2006', 'max']) -> xarray.DataArray:
    '''
    Converts .nc files to xarrays. 29 of February are excluded as in the .nc files.
    '''
    # path_to_data = os.path.join('..', 'data', 'stash', 'WindProject', 'cmip_stash')
    ncs = np.array(sorted(os.listdir(path_to_data)))
    ncs_filtered = [nc for nc in ncs if np.prod([cond in nc for cond in contains])]

    f1 = xarray.load_dataset(os.path.join(path_to_data, ncs_filtered[0]), decode_times=False)
    # Exclude 29 of February from date_range()
    first_time = pd.date_range(start=pd.to_datetime(ncs_filtered[0][-20:-12]), periods=f1.sizes['time'])
    t0 = np.apply_along_axis(check_leap_year, axis=0, arr=first_time)
    years_unique = np.unique(first_time[t0].year)
    second_time = pd.date_range(start=pd.to_datetime(ncs_filtered[0][-20:-12]), periods=f1.sizes['time'] + len(years_unique))
    for yea in years_unique:
        second_time = second_time.drop(pd.to_datetime(str(yea)+'-02-29'))
    f1['time'] = second_time
    f1_xarray = f1.to_array()
    # assert bool(np.prod([str(y) + '-02-29' in f1_xarray.time.loc[t0].data.astype('datetime64[D]').astype('str') for y in years_unique])), "no 29 feb on leap years"
    return f1_xarray


def reduce_to_area(data_arr: xarray.DataArray, lat_min: float = 41.12, lat_max: float = 81.49, lon_min: float = 19.38, lon_max: float = 169.40) -> xarray.DataArray:
    """
    Reduces the area to the input frames +-1.5 on latitude axis and +-2 on longitude axis.
    """
    band_name = [v for v in dict(data_arr.coords)['variable'].data if 'bnds' not in v][0]
    output = data_arr.sel(lat=slice(lat_min - 1.5, lat_max + 1.5), lon=slice(lon_min - 2, lon_max + 2), variable=band_name)
    output.data = np.float64(output.data)
    assert np.allclose(output.data[:, 0, :, :], output.data[:, 1, :, :]), "`bnds` dim components of tensor are not identical"
    return output


def interp_timewise(data_arr: xarray.DataArray, lon_res: float = 0.25, lat_res: float = 0.25, interp_method: str = 'linear', plot_example: bool =False) -> tuple:
    assert np.allclose(data_arr.data[:, 0, :, :], data_arr.data[:, 1, :, :]), "2d components of tensor are not identical"
    timesteps = len(data_arr.time.data)
    plot_example_dice = np.random.randint(0, timesteps)
    znews = []
    x = data_arr.lon.data
    y = data_arr.lat.data
    xnew = np.arange(x[0], x[-1], lon_res)
    ynew = np.arange(y[0], y[-1], lat_res)
    for t in tqdm(range(timesteps)):   
        # xx, yy = np.meshgrid(x, y)
        z = data_arr.data[t, 0, :, :]
        f = interpolate.interp2d(x, y, z, kind=interp_method)
        
        znew = f(xnew, ynew)
        znews.append(znew)
        if plot_example and t == plot_example_dice:
            plt.figure()
            plt.title('before')
            plt.imshow(z)
            plt.show()
            plt.figure()
            plt.title('after')
            plt.imshow(znew)
            plt.show()
    output = np.stack(znews)
    return (output, xnew, ynew)


def interp_timewise_xarray(data_arr: xarray.DataArray, lon_res: float = 0.25, lat_res: float = 0.25, interp_method: str = 'linear', plot_example: bool =False) -> xarray.DataArray:

    data_new, xnew, ynew = interp_timewise(data_arr, interp_method=interp_method, plot_example=plot_example)
    data_arr.coords
    coords_new = {'lon': ("lon", xnew), 'lat': ("lat", ynew), 'time': ("time", data_arr.time.data)}
    output = xarray.DataArray(data=data_new, coords=coords_new, dims=('time', 'lat', 'lon'), attrs=data_arr.attrs)

    return output


def get_file_paths(
    path_to_data: str = "drive/MyDrive/Belgorodskaya/*.tif",
    feature_names: list = ["tmax", "tmin", "pr"],
):
    """
    Filters out required features amongs terraclim dataset
    Arguments:
      path_to_data (str): path to directory that containts terraclim dataset
      feature_names (list): list of required features
    Returns:
      dict: key -- feature name; value -- list of related tif files
    """
    files_to_mosaic = glob.glob(path_to_data)
    files_to_mosaic = list(
        filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic)
    )
    file_paths = {
        fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names
    }
    return file_paths


def get_closest_pixel(dataset, coord):   # change gdal.Dataset to xarray
  """Finds the closest pixel indices in the dataset
  Args:
      dataset (gdal.Dataset): dataset with pixels
      coord (np.ndarray): coordinate for which the closest pixel's indices in the dataset will be found
  Returns:
      tuple: x, y indices among the dataset
  """
#   coords_dict = get_coords_res(dataset)
#   raster_xsize = dataset.RasterXSize
#   raster_ysize = dataset.RasterYSize
#   x_0, y_0, x_res, y_res = coords_dict['x'], coords_dict['y'], coords_dict['x_res'], coords_dict['y_res']
#   x_coords = np.array(range(raster_xsize)) * x_res + x_0    # x_coords = dataset.lat or lon
#   y_coords = np.array(range(raster_ysize)) * y_res + y_0    # y_coords = lon or lat    DO NOT CHANGE BELOW
  x_coords = dataset.lon
  y_coords = dataset.lat[::-1]
  R = 6371
  closest = 21212121
  I = 0
  J = 0
  for i, theta in enumerate(x_coords):
    for j, fi in enumerate(y_coords):
            r = great_circle((theta,fi), coord).kilometers
            if r < closest:
              closest = r
              I = i# + 1
              J = j# + 1
  closest_x_idx = I 
  closest_y_idx = J
  return closest_x_idx, closest_y_idx


def closest_pixel_for_station(station_name, dataset, station_list):   # Change gdal.Dataset to xarray
    """Finds the closest pixel indices in the dataset for the station
    Args:
      station_name (str): name of the station
      dataset (gdal.Dataset): dataset with pixels
      station_list (pd.DataFrame): table with stations' coordinates
    Returns:
      tuple: x, y indices among the dataset
    """
    station = station_list[station_list["Наименование станции"] == station_name]
    coord = [station["Долгота"].values[0], station["Широта"].values[0]]
    pix = get_closest_pixel(dataset=dataset, coord=coord)
    return pix
 

# def make_model_dataset(station_name: str,
#                        start_date: str,
#                        end_date: str,
#                        wind_cmip: np.array,
#                        station_list: pd.DataFrame,
#                        path_to_history: str='data/history', 
#                        path_to_elev: str='data/elev', 
#                        features: list=['tasmax', 'tasmin', 'pr'],
#                        pix: list=[-1, -1]):

#     table = pd.DataFrame({"Date":pd.date_range(start="01.01.2006",
#                                         end = "01.31.2020",
#                                         freq="D")})
#     for feature_name in features:
#         file_paths = [path_to_history+ '/' + fn for fn in os.listdir(path_to_history) if (fn[-4:] == '.tif' ) and feature_name in fn]   
#         dataset = gdal.Open(file_paths[0], gdal.GA_ReadOnly)
#         if pix[0] == -1 and pix[1] == -1:  
#           pix = closest_pixel_for_station(station_name=station_name, dataset=dataset, station_list=station_list)
#         array = []
#         for i in range(1, 5145):
#             band = dataset.GetRasterBand(i)
#             if feature_name in ["tasmax", "tasmin"]:
#                 arr = band.ReadAsArray()
#                 array.append(arr[pix[1]][pix[0]] - 273.15)
#             else:
#                 arr = band.ReadAsArray()
#                 array.append(arr[pix[1]][pix[0]])
#         table = table.join(pd.DataFrame({feature_name: array}))

#     array_wind = []
#     for day in wind_cmip:
#         array_wind.append(day[pix[1]][pix[0]])
#     table = table.join(pd.DataFrame({"CMIP_wind": array_wind}))

#     file_paths = [path_to_elev + "/" + "elevation.tif"]
#     dataset = gdal.Open(file_paths[0], gdal.GA_ReadOnly)
#     array = []
#     band = dataset.GetRasterBand(1)
#     arr = band.ReadAsArray()
#     array.append(arr[pix[1]][pix[0]])
#     table = table.join(pd.DataFrame({"el": 5144 * array}))
#     table = table.set_index("Date")
#     table = table.loc[table.index >= pd.to_datetime(start_date)]
#     table = table.loc[table.index <= pd.to_datetime(end_date)]
#     return table


# def get_coords_res(dataset: gdal.Dataset):
#     """
#     For given dataset returns position of top left corner and resolutions
#     Arguments:
#       dataset (osgeo.gdal.Dataset): gdal dataset
#     Returns:
#       dict: containts coordinates of top left corner and
#          resolutions alog x and y axes
#     """
#     gt = dataset.GetGeoTransform()
#     output = {}
#     output["x"] = gt[0]
#     output["y"] = gt[3]
#     output["x_res"] = gt[1]
#     output["y_res"] = gt[-1]
#     return output


# def extract_latitude_longtitute(path_to_tifs: str, feature_name: str):
#     """
#     Extract 1d arrays of longitutde and latitude of given .tif data
#     Helps to build a mapping between raster spatial indices and coordinates on the earth
#     Arguments:
#       path_to_data (str): path to directory that containts terraclim dataset
#       feature_names (str): feature name of the interest
#     Returns:
#       tuple: longtitude and latitude 1d arrays
#     """
#     file_paths = get_file_paths(path_to_tifs, [feature_name])
#     dset_tmp = gdal.Open(file_paths[feature_name][0])
#     coords_dict = get_coords_res(dset_tmp)
#     Lat = np.zeros((dset_tmp.RasterYSize))
#     for i in range(Lat.shape[0]):
#         Lat[i] = coords_dict["y"] + i * coords_dict["y_res"]
#     Lon = np.zeros((dset_tmp.RasterXSize))
#     for i in range(Lon.shape[0]):
#         Lon[i] = coords_dict["x"] + i * coords_dict["x_res"]
#     return Lat, Lon


# def plot_tl_positions(file_paths: list):
#     """
#     Viualize positions of top left corners of dataset given
#     Arguments:
#     file_paths (list): list of paths to files that contain datasets
#     """
#     tlxs = []
#     tlys = []
#     for fp in tqdm(file_paths):
#         dataset = gdal.Open(fp, gdal.GA_ReadOnly)
#         if dataset is not None:
#             coords_dict = get_coords_res(dataset)
#             tlxs.append(coords_dict["x"])
#             tlys.append(coords_dict["y"])

#     fig, ax = plt.subplots()
#     fig.set_figheight(15)
#     fig.set_figwidth(15)
#     ax.scatter(tlxs, tlys)

#     for i in range(len(tlxs)):
#         ax.annotate(i, (tlxs[i], tlys[i]))
#     plt.gca().set_aspect("equal", adjustable="box")
#     ax.set_title("Positions of top left corners of each raster")
#     ax.grid(True)


# def dataset_to_np(
#     dataset: gdal.Dataset, x_off: int, y_off: int, xsize: int, ysize: int, verbose: bool = False
# ):
#     """
#     Converts gdal.Dataset to numpy array
#     !NB: raster bands are enumerated starting from 1!
#     Arguments:
#       dataset (gdal.Dataset): dataset to cast
#       x_off (int): starting x position - idx
#       y_off (int): starting y position - idx
#       xsize (int): number of points to save in x direction
#       ysize (int): number of points to save in y direction
#     Returns:
#       np.ndarray -- 3d tensor of information given in dataset
#     """

#     shape = [dataset.RasterCount, ysize, xsize]
#     output = np.empty(shape)
#     if verbose:
#         for r_idx in tqdm(range(shape[0])):
#             band = dataset.GetRasterBand(r_idx + 1)
#             arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
#             output[r_idx, :, :] = np.array(arr)
#     else:
#         for r_idx in range(shape[0]):
#             band = dataset.GetRasterBand(r_idx + 1)
#             arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
#             output[r_idx, :, :] = np.array(arr)
#     return output


# def get_nps(feature_names, path_to_tifs, verbose=False, dset_num=0):
#     file_paths = get_file_paths(path_to_tifs, feature_names)
#     # open gdal files
#     dsets = {}
#     for fn in feature_names:
#         dset = gdal.Open(file_paths[fn][dset_num])
#         dsets[fn] = dset
#     # reading into np, scaling in accordance with terraclim provided
#     nps = {}
#     for fn in feature_names:
#         np_tmp = dataset_to_np(
#             dsets[fn],
#             x_off=0,
#             y_off=0,
#             xsize=dsets[fn].RasterXSize,
#             ysize=dsets[fn].RasterYSize,
#             verbose=verbose,
#         )
#         # Scaling in accordance with dataset description
#         if fn == "tmin" or fn == "tmax":
#             nps[fn] = np_tmp * 0.1
#         elif fn == "ws":
#             nps[fn] = np_tmp * 0.01
#         elif fn == "vap":
#             nps[fn] = np_tmp * 0.001
#         elif fn == "seasurfacetemp":
#             nps[fn] = np_tmp * 0.01
#         else:
#             nps[fn] = np_tmp

#     # getting mean temp if accessible
#     if "tmin" in feature_names and "tmax" in feature_names:
#         nps["tmean"] = (nps["tmax"] + nps["tmin"]) / 2

#     return nps


# def cmiper(cmip, lat_left, lat_right, lon_left, lon_right):
#     """
#     Preprocess CMIP to understandable DataFrame.
#     cmip - downloaded CMIP as a DataFrame.
#     """
#     a = cmip.loc[
#         (cmip.lat_bnds >= lat_left - 1.5) & (cmip.lat_bnds <= lat_right + 1.5)
#     ]  # Широта
#     a = a.loc[(a.lon_bnds >= lon_left) & (a.lon_bnds <= lon_right)]  # Долгота

#     a = a.reorder_levels(['time','bnds','lat','lon']).sort_index()  # Takes much time

#     a = a.reset_index()
#     a.drop(
#         columns=["time_bnds", "lat_bnds", "lon_bnds", "height", "bnds", "time"],
#         inplace=True,
#     )

#     lat_idx = list(set(a.lat))
#     lon_idx = list(set(a.lon))
#     le_lat, le_lon = preprocessing.LabelEncoder(), preprocessing.LabelEncoder()
#     le_lat.fit(lat_idx)
#     le_lon.fit(lon_idx)

#     a.insert(loc=0, column="lat_idx", value=le_lat.transform(a.lat))
#     a.insert(loc=1, column="lon_idx", value=le_lon.transform(a.lon))
#     a.drop(columns=["lat", "lon"], inplace=True)

#     return a


# def to_3dar(cmip_pr):
#     """
#     Preprocess CMIP to 3d np.array.
#     cmip_pr - preprocessed cmip.
#     """
#     day = int(cmip_pr.shape[0] / 3650)
#     lonn = cmip_pr.lon_idx.unique().shape[0]
#     latt = cmip_pr.lat_idx.unique().shape[0]
#     fin = []

#     for n in range(3650):
#         aa = cmip_pr[n * day : n * day + day]  # cannot put variable day here
#         z = np.zeros((latt, lonn))
#         for i, j, w in zip(aa["lat_idx"], aa["lon_idx"], aa["sfcWind"]):
#             z[i, j] = w
#         fin.append(z[1:-1])
#     fin = np.stack(fin)

#     x = np.arange(18, 170.01, 2)       # longitude   25
#     y = np.arange(39.75, 77.26, 1.5)  # latitude

#     xnew = np.arange(18, 170.01, 0.25)                         # 25
#     ynew = np.arange(39.75, 77.26, 0.25)#[::-1]              # 19           

#     ss_x = np.where(((xnew >= 18.75) & (xnew <= 169.25)))[0]   # 19 - 169
#     ss_y = np.where(((ynew >= 40.75) & (ynew <= 77.25)))[0]    # 41 - 77

#     final = []
#     for z in fin:
#         f = interpolate.interp2d(x, y, z)  # or (x, y, z)
#         final.append(f(xnew, ynew)[np.ix_(ss_y, ss_x)])
#     final = np.stack(final)

#     new_final = []
#     for i in final:
#         new_final.append(i[::-1])
#     final = np.stack(new_final)

#     return final


# def extract_cmip_grid(cmip, lat_left, lat_right, lon_left, lon_right):
#     """
#     Consist of all needed functions for the preprocessing of CMIP.
#     """

#     intermediate = cmiper(cmip, lat_left, lat_right, lon_left, lon_right)
#     return to_3dar(intermediate)


def leap_years(ar, leap_idx):
    """
    Inserts 29 of February in a 3d-array.
    leap_idx - indices of 29 of February in descending order.
    """
    for i in leap_idx:
        feb = (ar[i] + ar[i + 1]) / 2
        feb = np.expand_dims(feb, axis=0)
        ar = np.vstack((ar[:i+1], feb, ar[i+1:]))
    
    return ar
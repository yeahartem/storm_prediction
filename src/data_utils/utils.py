import numpy as np
from matplotlib import pyplot as plt
from calendar import isleap
# import gdown
import xarray

# def download_gdrive(gdrive_folder_link: str) -> None:
#   """Downloads the files in google drive folder into current directory

#   Args:
#       gdrive_folder_link (str): link to google drive folder
#   """
#   gdown.download_folder(gdrive_folder_link)

def check_leap_year(date):
  
  year = date.astype('datetime64[Y]').astype(int) + 1970
  
  return np.logical_not(np.logical_and(np.not_equal(year % 4, 0),  np.logical_or(np.not_equal(year % 100, 0), np.equal(year % 400, 0))))

def test_check_leap_year(tmp: xarray.DataArray):
  
  # check_leap_year(tmp.time.data)
  # tmp.time.data
  t0=np.apply_along_axis(check_leap_year, axis=0, arr=tmp.time.data)
  t1=np.array([isleap(v) for v in (tmp.time.data.astype('datetime64[Y]').astype(int) + 1970)])
  assert (t1 == t0).all(), "check_leap_year does not work correctly"    

def deg_min_to_dec(degrees, minutes):
  return degrees + minutes / 60.

# def get_htc(tmean, pr, plot=True, x_pos=0, y_pos=0):
#   """
#   Computes Hydro-thermal coefficient for 3d arrays of tmean and pr
  
#   Arguments:
#     tmean (3darray): mean temperatures monthly
#     pr (3darray): total precipitations monthly
#     plot (bool): if to plot an example at coords x_pos, y_pos
#     x_pos (int): latitude index
#     y_pos (int): longitude index
#   """
#   #mask for months with temp greater than 10 degees C
#   tmask = tmean >= 10

#   tmean_masked = tmean * tmask
#   pr_masked = pr * tmask

#   htc = np.empty((pr_masked.shape[0] // 12, pr_masked.shape[1], pr_masked.shape[2]))
#   for i in range(htc.shape[0]):
#     htc[i, :, :] =  10 * pr_masked[12 * i : 12 * (i + 1), :, :].sum(axis = 0) / (30 * tmean_masked[12 * i : 12 * (i + 1), :, :].sum(axis = 0))
#   if plot:
#     plt.figure(figsize=(16, 9))

#     plt.plot(htc[:, x_pos, y_pos])
#     plt.legend()
#     plt.ylabel('HTC value')
#     plt.xlabel('Year')
#     plt.grid()

#   return htc
import numpy as np
from matplotlib import pyplot as plt
from calendar import isleap
import xarray


def check_leap_year(date):
  
  year = date.astype('datetime64[Y]').astype(int) + 1970
  
  return np.logical_not(np.logical_and(np.not_equal(year % 4, 0),  np.logical_or(np.not_equal(year % 100, 0), np.equal(year % 400, 0))))

def test_check_leap_year(tmp: xarray.DataArray):
  
  
  t0=np.apply_along_axis(check_leap_year, axis=0, arr=tmp.time.data)
  t1=np.array([isleap(v) for v in (tmp.time.data.astype('datetime64[Y]').astype(int) + 1970)])
  assert (t1 == t0).all(), "check_leap_year does not work correctly"    

def deg_min_to_dec(degrees, minutes):
  return degrees + minutes / 60.


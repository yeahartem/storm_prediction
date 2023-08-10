import numpy as np
import os


mean_channels_cmip6 = [8.934635,     # sfcWindmax
                       6.9155507,     # sfcWind
                       2.8345312e-05, # pr
                       100936.055,    # psl
                       280.37646,     # tasmax
                       276.80447,]    # tasmin

std_channels_cmip6 = [4.7078495,
                       3.769951,
                       7.100028e-05,
                       1362.2008,
                       20.841803,
                       20.852707]

mean_channels_cmip5 = [9.60365104675293,        # sfcWindmax
                       7.4015727043151855,      # sfcWind
                       2.7782780307461508e-05,  # pr
                       100915.8984375,          # psl
                       280.5318908691406,       # tasmax
                       276.5341491699219,]      # tasmin

std_channels_cmip5 =  [5.886378288269043,
                       4.558711051940918,
                       7.818317681085318e-05,
                       1514.74365234375,
                       20.955684661865234,
                       21.78408432006836]

elevation_mean = 377.73032
elevation_std =  855.89075
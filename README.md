
# TO BE UPDATED 

# Environments:
`wind_env` - `environments/requirements.txt` -- most recent environment, better to use it.

# Docker:
From repo folder run:
* `docker build -t wind_dev .`
* `docker run -it  -v  <CODE FOLDER>:/wind -v <DATA FOLDER>:/wind/data_mounted -m 32000m  --cpus=8  --gpus '"device=2"' -w="/wind" wind_dev`

May need more than 64 Gb of RAM

# Data, NN
Expected data folder structure:
``` bash
wind_data2
├── cmip
│   ├── elevation
│   │   └── elevation.nc
│   ├── pr_day_inmcm4_rcp45_r1i1p1_20060101-20151231.nc
│   ├── pr_day_inmcm4_rcp45_r1i1p1_20160101-20251231.nc
│   ├── sfcWind_day_inmcm4_rcp45_r1i1p1_20060101-20151231.nc
│   ├── sfcWind_day_inmcm4_rcp45_r1i1p1_20160101-20251231.nc
│   ├── tasmax_day_inmcm4_rcp45_r1i1p1_20060101-20151231.nc
│   ├── tasmax_day_inmcm4_rcp45_r1i1p1_20160101-20251231.nc
│   ├── tasmin_day_inmcm4_rcp45_r1i1p1_20060101-20151231.nc
│   └── tasmin_day_inmcm4_rcp45_r1i1p1_20160101-20251231.nc
└── weather_stations
    ├── data_meteo_full.csv
    ├── data_meteo_full.parquet
    ├── weatherstation_list.csv
    └── weatherstation_list.json
```
* `data_meteo_full.parquet` may be created from `data_meteo_full.csv` using `src/data_utils/parquet.py` script
* `weatherstation_list.json` and `weatherstation_list.csv` different from the version used before!

* Run `pipeline/train.py` to generate data and train NN model (see `conf/train_conf.json` for parameters)
* Run `pipeline/infer.py` to generate data to infer NN model on (see `conf/train_conf.json` for parameters, including path to pretrained NN)

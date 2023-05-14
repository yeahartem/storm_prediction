

# Run train:

Train regression:

* `python src/regression/train.py --config-path <CONFIG>`

# Docker:

From repo folder run:
* `docker build -t wind_dev .`
* `docker run -it  -v  <CODE FOLDER>:/wind -v <DATA FOLDER>:/wind/data_mounted -m 32000m  --cpus=8  --gpus '"device=2"' -w="/wind" wind_dev`

Example:
* `docker run -it  -v  $(pwd)/Wind:/wind -v $(pwd)/data:/wind/data_mounted -m 64000m  --cpus=16  --gpus '"device=0,1"' -w="/wind" wind_dev`

* `mkdir $(pwd)/wind`
* `chown 101:101 $(pwd)/Wind`



May need more than 64 Gb of RAM

# Configs 

Configs handeled by [hydra](https://hydra.cc/docs/intro/) library. Configs are located in `configs` folder.


# Environments:

For local run:

`environments/environment.yml` -- most recent environment

To install run:

`conda env create --name wind_env --file=environments/environment.yml`


# Data format:

Binary target expects:
* Climate data in netcdf format. One file per climate variable.
* Target data in parquet format, 'y' column is the target, where values is float. (binarization happens in src/binary_target/datamodule prepare_data function)
* Normalization values in .npy format, std and mean separately


# How to prepare data from source:

* `data_meteo_full.parquet` may be created from `data_meteo_full.csv` using `src/data_utils/parquet.py` script
* `weatherstation_list.json` should be downloaded separately
*  CMIP climate data files in netcdf format
*  NOAA meteo staions data in parquet format
* `src/data_assemble/prepare_cmip5.py` script should be run to prepare data for training or inferring

## Infer
* Run `/wind/src/data_assemble/prepare_cmip5.py` (row `110` - select time limits) to prepare data for training or inferring
* Run `/wind/src/binary_target/infer.py`. It has config file `/wind/configs/infer_conf.json`
* Output will be saved into `os.path.join(conf.path_to_save, conf.region_name)`, where `conf` is the config file from the bullet above.
* Here you will find `output.parquet` - geopandas with output probabilities by date and coordinate, `pics` folder with sample visualization
* Visualization -- rows 117-132 in `/wind/src/binary_target/infer.py`
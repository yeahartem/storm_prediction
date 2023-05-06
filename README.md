# Environments:
`wind_env` - `environments/requirements.txt` -- most recent environment, better to use it.

# Docker:
From repo folder run:
* `docker build -t wind_dev .`
* `docker run -it  -v  <CODE FOLDER>:/wind -v <DATA FOLDER>:/wind/data_mounted -m 32000m  --cpus=8  --gpus '"device=2"' -w="/wind" wind_dev`

Example:
* `docker run -it  -v  $(pwd)/wind:/wind -v $(pwd)/data:/wind/data_mounted -m 64000m  --cpus=16  --gpus '"device=0,1"' -w="/wind" wind_dev`

May need more than 64 Gb of RAM

# Configs 

Configs handeled by [hydra](https://hydra.cc/docs/intro/) library. Configs are located in `configs` folder.


# Data, NN

* `data_meteo_full.parquet` may be created from `data_meteo_full.csv` using `src/data_utils/parquet.py` script
* `weatherstation_list.json` and `weatherstation_list.csv` different from the version used before!

# How to train and infer
## Train
* Run `/wind/src/data_assemble/prepare_cmip5.py` (row `110` - select time limits) to prepare data for training or inferring
* NB: training time limits are limited with weather stations measurements available
* Run `/wind/src/binary_target/train.py`. It has config file `/wind/configs/train_conf.json`
## Infer
* Run `/wind/src/data_assemble/prepare_cmip5.py` (row `110` - select time limits) to prepare data for training or inferring
* Run `/wind/src/binary_target/infer.py`. It has config file `/wind/configs/infer_conf.json`
* Output will be saved into `os.path.join(conf.path_to_save, conf.region_name)`, where `conf` is the config file from the bullet above.
* Here you will find `output.parquet` - geopandas with output probabilities by date and coordinate, `pics` folder with sample visualization
* Visualization -- rows 117-132 in `/wind/src/binary_target/infer.py`
# Configs
Configs handeled by [hydra](https://hydra.cc/docs/intro/) library. Configs are located in `configs` folder.
`configs` folder contain **model** general config that includes information about model, processing and training steps to be taken. The user is supposed to manipulate experiment setup (training, inferring, preprocessing) using this **model** general config, e.g., `configs/train_WindNetElev83x41_test_run.yaml`.
## Model
* `model_name` -- model from `src/regression/models/models.py`
* `time_window` -- temporal receptive field
* `half_side_size` -- spatial receptive field
## Processing 
This part uses `configs/raw` and `configs/process` folders. The former deals with raw data and the latter deals with processing steps with prescribed attributes.
* `raw` -- paths to raw data: CMIP, elevation, **raw** and parsed weather stations data
* `process` -- processing logic. See comments in `configs/process/cmip6_elevation_dataset.yaml` config.

## Training
This part uses `configs/train` folder. This part describes training and datamodule logic. See comments in `configs/train/train_WindNetElev83x41_test_run.yaml` config.

# Data preparation:
This step must be taken prior to **both** training (`train.py`) and inferring (`run.py`). Config for `preprocess.py` must be taken from `configs` folder. An example is `configs/train_WindNetElev83x41_test_run.yaml`. See Section **Configs** for reference.

This step creates folder under the name specified in `cfg.process.data_dir` attribute. This folder will contain coordinates, preprocessed data and **preprocessed target data** in `target.parquet`, `target.parquet.pp1`, `target.parquer.pp2`. See **Configs** section for reference. Further, preprocessed data from this folder will be used to be assembled into data required for training and inferring.
```
python preprocess.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python preprocess.py --config-path /app/wind/configs --config-name train_WindNetElev83x41_test_run
```
# Run train:
**Prior to training run `preprocess.py`** to prepare data!
To specify parameters, please be referred to **Configs** section.
Train regression:
```
python train.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python run.py --config-path configs --config-name train_WindNetElev83x41_test_run.yaml
```

# Run inference: SEVA, PLEASE UPDATE THAT ONCE YOU FINISH WITH INFER REWORKING!
**Prior to inferring run `preprocess.py`** to prepare data!
For  
```
python run.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python run.py --config-path infer_configs/ --config-name **Before training run `preprocess.py`** to prepare data!
```
# Docker:

From repo folder run:

```
docker build -t .

export WANDB_API_KEY=<key>

docker run -it \
   -v $(pwd):/app/wind \
   -v <DATA FOLDER>:/app/wind/data \
   -m 128000m --cpus=16 --gpus '"device=0,1"' \
   --ipc=host \
   --user="$(id -u):$(id -g)" \
   -w="/app/wind" \
   -e "WANDB_API_KEY=$WANDB_API_KEY" \
   -e "WANDB_DATA_DIR=/app/wind/out" \
   -e "WANDB_DIR=/app/wind/out" \
   -e "WANDB_CACHE_DIR=/app/wind/out" \
   wind_dev116

```
Example:


```
   docker run -it \
   -v $(pwd):/app/wind \
   -v /mnt/data/lukashevich/:/app/wind/data \
   -m 128000m --cpus=16 --gpus '"device=0,1"' \
   --ipc=host \
   --user="$(id -u):$(id -g)" \
   -w="/app/wind" \
   -e "WANDB_API_KEY=$WANDB_API_KEY" \
   -e "WANDB_DATA_DIR=/app/wind/out" \
   -e "WANDB_DIR=/app/wind/out" \
   -e "WANDB_CACHE_DIR=/app/wind/out" \
   wind_dev116

```

# Data format:

Binary target expects:
* Climate data in netcdf format. One file per climate variable.
* Target data in parquet format, 'y' column is the target, where values is float.
* Normalization values in .npy format, std and mean separately (use only 32 precision values)


# How to prepare data from source:

* `data_meteo_full.parquet` may be created from `data_meteo_full.csv` using `src/data_utils/parquet.py` script
* `weatherstation_list.json` should be downloaded separately
*  CMIP climate data files in netcdf format
* `preprocess.py` script should be run to prepare data for training or inferring

## Infer
* Run `preprocess.py` to prepare data for training or inferring
* Run `run.py`. It has config file like `configs/infer_configs/infer_world_reg_test.taml`
* Output will be saved into `out`
* Here you will find `.kml` file with estimated risks. Also, there will be raw inference from which you may want to get any statistic
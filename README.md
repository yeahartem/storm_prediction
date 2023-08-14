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

This step creates folder under the name specified in `cfg.process.data_dir` attribute. This folder will contain coordinates, preprocessed data and **preprocessed target data** in `target.parquet`, `target.parquet.pp1`, `target.parquet.pp2`. See **Configs** section for reference. Further, preprocessed data from this folder will be used to be assembled into data required for training and inferring.
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

# Run inference:
**Prior to inferring run `preprocess.py`** to prepare data!
Configure evalutaion process by providing a path to eval config if general config in `configs` folder, e.g., `configs/cmip6_elevation_WindNetElev83x41.yaml`, attribute `defaults.eval`.
In `configs/eval` configuration file you can select inferring period, or pass it through command line as in example below. See Hydra docs for reference https://hydra.cc/docs/advanced/override_grammar/basic/
```
python run.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python run.py --config-path configs/ --config-name cmip5_WindNet41x41.yaml time_start='2019-01-30' time_end='2019-01-31'
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


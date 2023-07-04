# Data preparation:
```
python preprocess.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python preprocess.py --config-path /app/wind/configs/dataset_configs/ --config-name cmip6_dataset_world_infer_test
```
# Run train:
To prepara data for training go to Data preparation section
Train regression:
```
python train.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python run.py --config-path /app/wind/configs/train_configs/ --config-name train_world_reg_test
```

# Run inference:
To prepara data for inference go to Data preparation section. You should have firstly have a) prepared dataset for training -- needed for normalization values and b) trained model
```
python run.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Example:
```
python run.py --config-path /app/wind/configs/infer_configs/ --config-name infer_world_reg_test
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


# Configs 

Configs handeled by [hydra](https://hydra.cc/docs/intro/) library. Configs are located in `configs` folder.





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
# Environments:
* `wind_env` - `environments/environment.yml` -- main environment, better to use it.
# Data, NN
* Run `pipeline/train.py` to generate data and train NN model (see `conf/train_conf.json` for parameters)
* Run `pipeline/infer.py` to generate data to infer NN model on (see `conf/train_conf.json` for parameters, including path to pretrained NN)

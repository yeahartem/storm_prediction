import sys,os
sys.path.append(os.getcwd())
import logging
import warnings
import random
from omegaconf import DictConfig, OmegaConf
import time
import hydra

import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point
import geopandas as gpd
import fiona
from fiona.drvsupport import supported_drivers
supported_drivers['LIBKML'] = 'rw'

warnings.filterwarnings("ignore")
random.seed(112)

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')


def risk_estimation(cfg: DictConfig) -> None:        
    start_time = time.process_time()  
    logging.info(f"Reading raw inference")
    df_infer = pd.read_csv(os.path.join(cfg.path_to_predictions, "result.csv"))

    logging.info(f"Grouping by months, estimating risk")
    df_infer['date'] = pd.to_datetime(df_infer['date'])
    df_grpby = df_infer.groupby(['lat', 'lon', df_infer.date.dt.year, df_infer.date.dt.month])
    df_risks = df_grpby['prediction'].agg(lambda x: (x > cfg.wind_risk_threshold).mean())

    logging.info(f"Preparing format for .kml dumping")
    df_risks_ = df_risks.index.rename(['lat', 'lon', 'year', 'month']).to_frame().reset_index(drop=True)
    df_risks_['prob'] = df_risks.values
    df_risks = df_risks_
    df_risks['day'] = np.ones(len(df_risks))
    df_risks['timestamp'] = pd.to_datetime(df_risks[['year', 'month', 'day']])
    df_risks = df_risks.drop(columns=['year', 'month', 'day'])
    logging.info(f"Casting to geopandas")
    df = df_risks
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    gdf = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    fiona.supported_drivers['KML'] = 'rw'
    gdf.to_file(cfg.output_file, driver='KML')
    logging.info(f"Saved to {cfg.output_file}")
    logging.info(f"Total time spent {time.process_time() - start_time} seconds")



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="risk_estimation_20_test")
def main(cfg: DictConfig):    
    risk_estimation(cfg)
    logging.info('Risks are estimated finished!')


if __name__ == "__main__":      
    main()
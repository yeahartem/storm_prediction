import numpy as np
from omegaconf import DictConfig, OmegaConf
import os 
import pandas as pd
import polars




def load_target(cfg):
    """Load target data from given folder"""
    target_df = polars.read_parquet(cfg.path_to_prepared_target_data)
    target_df = (
        target_df
        .lazy()        
        .sort("time")
        .groupby(["station_name"])
        .agg(
            [polars.col('time'), polars.col('y')]
        )
        .collect()
    )

    return target_df


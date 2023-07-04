import pandas as pd

df = pd.read_csv("./data/weather_stations/data_meteo_full.csv", engine="pyarrow")
df.to_parquet("./data/weather_stations/data_meteo_full.parquet", compression=None)

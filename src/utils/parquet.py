import pandas as pd

df = pd.read_csv("./data_mounted/weather_stations/data_meteo_full.csv", engine="pyarrow")
df.to_parquet("./data_mounted/weather_stations/data_meteo_full.parquet", compression=None)

import sys
import os
sys.path.append('..')

from data_utils.utils import download_gdrive

elevation_url = 'https://drive.google.com/drive/folders/1q0uZfQJ5Lo4rJFtSkBh9SmEUTjC_byh5?usp=sharing'
history_climate_url = 'https://drive.google.com/drive/folders/1oCpAnqDj5RrFIR7MOFtuy3LenINc1wkz?usp=sharing'
future_climate_url = 'https://drive.google.com/drive/folders/1fHgH-NA4B4stIgVbZd-zv8aR190pds_Z?usp=sharing'

urls_dict = {"elev": elevation_url, "history": history_climate_url, "future": future_climate_url}

if __name__ == "__main__":
    for (k, v) in urls_dict.items():
        if not os.path.isdir(k):
            os.mkdir(k)
    
        os.chdir(k)
        print("Downloading ", k)
        download_gdrive(v)
        os.chdir('../..')
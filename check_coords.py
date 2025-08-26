import xarray as xr
import os
import warnings

# Подавляем предупреждения, чтобы вывод был чище
warnings.filterwarnings("ignore")

# --- Укажите имя одного из ваших .nc файлов ---
filename_to_check = 'psl_day_cmip5_cmip5world_r1i1p1_1950-01-01-2020-11-13.nc'
# ---------------------------------------------

try:
    data_path = os.path.join('data', 'cmip5_orig', filename_to_check)
    ds = xr.open_dataset(data_path)

    print("="*40)
    print(f"Анализ файла: {filename_to_check}")
    print("="*40)
    print(f"Диапазон времени (time):")
    print(f"  Мин: {ds.time.min().values}")
    print(f"  Макс: {ds.time.max().values}")
    print("\nДиапазон широты (lat):")
    print(f"  Мин: {ds.lat.min().values}")
    print(f"  Макс: {ds.lat.max().values}")
    print("\nДиапазон долготы (lon):")
    print(f"  Мин: {ds.lon.min().values}")
    print(f"  Макс: {ds.lon.max().values}")
    print("="*40)

except FileNotFoundError:
    print(f"ОШИБКА: Файл не найден по пути: {data_path}")
    print("Пожалуйста, убедитесь, что имя файла в скрипте верное и он лежит в папке data/cmip5_orig/")
except Exception as e:
    print(f"Произошла ошибка: {e}")
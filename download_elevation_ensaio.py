import ensaio
import xarray as xr
import shutil, os

print('Downloading earth topography via ensaio (~45 MB)...')
path = ensaio.fetch_earth_topography(version=1)
print(f'Downloaded to: {path}')

ds = xr.open_dataset(path)
print(ds)

# Copy to data/elevation.nc
os.makedirs('data', exist_ok=True)
dest = 'data/elevation.nc'
shutil.copy(path, dest)
print(f'Copied to {dest}')

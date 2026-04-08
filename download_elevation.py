"""
Download ETOPO1 global elevation data from NOAA and save as data/elevation.nc

ETOPO1 is a 1 arc-minute global relief model (~450 MB uncompressed).
After downloading, run preprocess.py with make_elevation_data=true to
regrid it to the CMIP6 grid and save as data/cmip6_world/elev_16.npy.
"""
import os
import gzip
import shutil
import urllib.request

URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.nc.gz"
GZ_PATH = "data/elevation.nc.gz"
NC_PATH = "data/elevation.nc"

def download_with_progress(url, dest):
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            mb_done = count * block_size / 1024**2
            mb_total = total_size / 1024**2
            print(f"\r  {pct}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    if os.path.exists(NC_PATH):
        print(f"Already exists: {NC_PATH}")
    else:
        print(f"Downloading ETOPO1 from NOAA (~75 MB compressed)...")
        print(f"  URL: {URL}")
        download_with_progress(URL, GZ_PATH)
        print(f"Extracting to {NC_PATH} ...")
        with gzip.open(GZ_PATH, "rb") as f_in:
            with open(NC_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(GZ_PATH)
        print(f"Done. Saved to {NC_PATH}")
        import os
        size_mb = os.path.getsize(NC_PATH) / 1024**2
        print(f"File size: {size_mb:.0f} MB")

    print("\nNext step:")
    print("  Set make_elevation_data: true in configs/process/cmip6_world_elevation_dataset.yaml")
    print("  Run: python preprocess.py --config-name=cmip6_world_elevation")

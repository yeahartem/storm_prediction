"""
inference_map.py  —  Figure 6-style storm probability maps

Generates probability heat-maps for a given region and 1-3 dates using a
trained GhostWindNet27 checkpoint.  No Hydra / DataModule needed.

Usage examples
--------------
# Hurricane Milton, Florida landfall October 9 2024 (3 panels):
python inference_map.py \
    --ckpt out/2026-04-09/15-37-42/epoch=12-step=65000.ckpt \
    --dates 2024-09-25,2024-10-09,2024-10-23 \
    --output fig_milton.png

# Single date, custom region:
python inference_map.py \
    --ckpt out/.../epoch=12.ckpt \
    --dates 2024-10-09 \
    --lat_min 15 --lat_max 40 --lon_min -100 --lon_max -60 \
    --output fig_single.png
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

sys.path.append(os.getcwd())
from src.utils.data_utils import make_padding
from src.regression.models.models import GhostWindNet27

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants matching training config
# ---------------------------------------------------------------------------
DATA_DIR   = "data/cmip6_world"
HALF_SIDE  = 47            # half_side_size in config → patch is 95×95
TIME_WIN   = 27            # time_window in config
PRECISION  = 16
VARIABLES  = ["sfcWindmax", "pr", "tasmax", "tasmin"]
BATCH_SIZE = 64            # cells per forward pass, keep low for 8 GB GPU


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_coords():
    lat = np.load(os.path.join(DATA_DIR, "lat.npy"))       # (160,)
    lon = np.load(os.path.join(DATA_DIR, "lon.npy"))       # (320,)
    time = np.load(os.path.join(DATA_DIR, "time.npy")).astype("datetime64[D]")  # (9124,)
    return lat, lon, time


def date_to_idx(date_str, time_arr):
    """Return index of the closest day in time_arr to the given date string."""
    target = np.datetime64(date_str, "D")
    idx = int(np.argmin(np.abs(time_arr - target)))
    actual = str(time_arr[idx])
    log.info(f"Requested {date_str} → index {idx} ({actual})")
    return idx


def build_padded_slice(time_idx, elev_padded, use_elevation=True):
    """
    Load 27 days of CMIP6 data (memory-mapped), apply global make_padding.
    If use_elevation, append elevation as 5th channel.

    Returns:
        combined – np.ndarray float32  (4 or 5, 27, 254, 414)
        shift    – tuple (47, 47)
    """
    half = TIME_WIN // 2                          # 13
    t0, t1 = time_idx - half, time_idx + half + 1  # [t0, t1)

    slices = []
    for var in VARIABLES:
        arr = np.load(os.path.join(DATA_DIR, f"{var}_{PRECISION}.npy"), mmap_mode="r")
        slices.append(arr[t0:t1].astype(np.float32))   # (27, 160, 320)
    data = np.stack(slices, axis=0)                    # (4, 27, 160, 320)

    # apply spherical padding (wraps lon, reflects lat at poles)
    data_padded, shift = make_padding(data, HALF_SIDE)  # (4, 27, 254, 414)

    if not use_elevation:
        return data_padded, shift

    # elevation channel — static, broadcast across time; elev_padded is (254, 414)
    elev_t = np.broadcast_to(
        elev_padded[np.newaxis, np.newaxis],            # (1, 1, 254, 414)
        (1, TIME_WIN) + elev_padded.shape               # (1, 27, 254, 414)
    ).copy()

    combined = np.concatenate([data_padded, elev_t], axis=0)  # (5, 27, 254, 414)
    return combined, shift


def compute_pos(date_np64, lat_deg, lon_deg):
    """
    Return positional encoding vector matching pixel_aggregation in data_load.py.
    Output: (4,) = [time_pos, time_pos_m, lat_pos, lon_pos]
    """
    d = date_np64.astype(object)
    month = d.month
    day   = d.day
    time_pos   = (month * 30.5 + day) / 365.0
    time_pos_m = month / 12.0
    lat_pos    = lat_deg / 90.0
    lon_pos    = lon_deg / 180.0
    return np.array([time_pos, time_pos_m, lat_pos, lon_pos], dtype=np.float32)


def region_cells(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Return list of (lat_orig_idx, lon_orig_idx, lat_deg, lon_deg)
    for all CMIP6 grid cells inside the specified region.
    Handles both [-180,180] and [0,360] lon conventions automatically.
    """
    cells = []
    for li, la in enumerate(lat):
        if la < lat_min or la > lat_max:
            continue
        for lj, lo in enumerate(lon):
            # normalise to [-180,180] for comparison
            lo_norm = lo if lo <= 180 else lo - 360
            if lo_norm < lon_min or lo_norm > lon_max:
                continue
            cells.append((li, lj, float(la), float(lo_norm)))
    return cells


def run_inference(model, device, combined_padded, cells, date_np64, in_chans=5):
    """
    Run model on all cells and return list of (lat_deg, lon_deg, prob).
    combined_padded: np.ndarray (in_chans, 27, 254, 414) float32
    """
    results = []
    n = len(cells)
    expected_shape = (in_chans, TIME_WIN, 95, 95)
    for start in range(0, n, BATCH_SIZE):
        batch_cells = cells[start : start + BATCH_SIZE]
        Xs, poses = [], []
        for (li, lj, la_deg, lo_deg) in batch_cells:
            # padded indices
            pl = li + HALF_SIDE
            ql = lj + HALF_SIDE
            x = combined_padded[
                :,
                :,
                pl - HALF_SIDE : pl + HALF_SIDE + 1,
                ql - HALF_SIDE : ql + HALF_SIDE + 1,
            ]   # (in_chans, 27, 95, 95)
            if x.shape != expected_shape:
                log.warning(f"Skip cell ({li},{lj}): unexpected patch shape {x.shape}")
                continue
            Xs.append(x)
            pos_vec = compute_pos(date_np64, la_deg, lo_deg)  # (4,)
            poses.append(np.tile(pos_vec, (TIME_WIN, 1)))      # (27, 4)

        if not Xs:
            continue

        X_t   = torch.from_numpy(np.stack(Xs)).to(device)    # (B, 5, 27, 95, 95)
        pos_t = torch.from_numpy(np.stack(poses)).to(device)  # (B, 27, 4)

        with torch.no_grad():
            logits = model([X_t, pos_t]).squeeze(1)           # (B,)
            probs  = torch.sigmoid(logits).cpu().numpy()

        for i, (li, lj, la_deg, lo_deg) in enumerate(batch_cells[:len(probs)]):
            results.append((la_deg, lo_deg, float(probs[i])))

        log.info(f"  Cells {start}-{start+len(batch_cells)}/{n}  "
                 f"prob range [{probs.min():.3f}, {probs.max():.3f}]")

    return results


def results_to_grid(results, lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Convert flat results list to 2-D probability grid (lon in [-180,180])."""
    lon_norm = np.where(lon <= 180, lon, lon - 360)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon_norm >= lon_min) & (lon_norm <= lon_max)
    lat_sub = lat[lat_mask]
    lon_sub = lon_norm[lon_mask]

    grid = np.full((len(lat_sub), len(lon_sub)), np.nan, dtype=np.float32)
    for (la, lo, p) in results:
        li = np.argmin(np.abs(lat_sub - la))
        lj = np.argmin(np.abs(lon_sub - lo))
        grid[li, lj] = p

    return grid, lat_sub, lon_sub


def _shapefile_lines(shp_path, lon_min, lon_max, lat_min, lat_max):
    """Read a shapefile with pyshp and return list of (lons, lats) line segments clipped to bbox."""
    import shapefile
    lines = []
    try:
        sf = shapefile.Reader(shp_path)
        for shape in sf.shapes():
            pts = np.array(shape.points)
            if len(pts) == 0:
                continue
            # quick bbox check
            if pts[:, 0].max() < lon_min or pts[:, 0].min() > lon_max:
                continue
            if pts[:, 1].max() < lat_min or pts[:, 1].min() > lat_max:
                continue
            # split by part indices
            parts = list(shape.parts) + [len(pts)]
            for a, b in zip(parts[:-1], parts[1:]):
                seg = pts[a:b]
                lines.append((seg[:, 0], seg[:, 1]))
    except Exception as e:
        log.warning(f"Could not read {shp_path}: {e}")
    return lines


_CARTOPY_CACHE = os.path.expanduser(r"~\.local\share\cartopy\shapefiles\natural_earth")


def _get_shapefiles():
    """Return paths to Natural Earth shapefiles from cartopy local cache."""
    # try 50m first, fall back to 110m
    def pick(*candidates):
        for c in candidates:
            p = os.path.join(_CARTOPY_CACHE, c)
            if os.path.exists(p):
                return p
        return None

    coast     = pick("physical/ne_50m_coastline.shp",
                     "physical/ne_110m_coastline.shp")
    countries = pick("cultural/ne_50m_admin_0_countries.shp",
                     "cultural/ne_110m_admin_0_countries.shp")
    states    = pick("cultural/ne_50m_admin_1_states_provinces_lines.shp",
                     "cultural/ne_110m_admin_1_states_provinces_lines.shp")

    if coast:
        log.info(f"Shapefiles: coast={os.path.basename(coast)}, "
                 f"countries={os.path.basename(countries) if countries else 'none'}, "
                 f"states={os.path.basename(states) if states else 'none'}")
    else:
        log.warning("No shapefiles found — no borders will be drawn")
    return coast, countries, states


def plot_maps(grids_dates, output_path, title="GhostWindNet27 — Storm Probability"):
    """
    grids_dates: list of (prob_grid, lat_arr, lon_arr, date_str, storm_lon, storm_lat)
    """
    shp_coast, shp_countries, shp_states = _get_shapefiles()

    n = len(grids_dates)
    fig = plt.figure(figsize=(6 * n, 5))
    gs  = GridSpec(1, n, figure=fig, wspace=0.08)

    # Dynamic colorscale: 2nd–98th percentile across all panels
    all_vals = np.concatenate([g[~np.isnan(g)] for g, *_ in grids_dates])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, (grid, lat_arr, lon_arr, date_str, storm_lon, storm_lat) in enumerate(grids_dates):
        lon_min_ext = lon_arr.min() - 0.5
        lon_max_ext = lon_arr.max() + 0.5
        lat_min_ext = lat_arr.min() - 0.5
        lat_max_ext = lat_arr.max() + 0.5

        ax = fig.add_subplot(gs[i])

        # probability heatmap
        ax.pcolormesh(lon_arr, lat_arr, grid, cmap=cmap, norm=norm, zorder=1)

        # draw borders using pyshp (pure Python, no GDAL needed)
        bbox = (lon_min_ext, lon_max_ext, lat_min_ext, lat_max_ext)
        if shp_states:
            for lons, lats in _shapefile_lines(shp_states, *bbox):
                ax.plot(lons, lats, color="#888", lw=0.35, zorder=2)
        if shp_countries:
            for lons, lats in _shapefile_lines(shp_countries, *bbox):
                ax.plot(lons, lats, color="#333", lw=0.5, zorder=3)
        if shp_coast:
            for lons, lats in _shapefile_lines(shp_coast, *bbox):
                ax.plot(lons, lats, color="black", lw=0.8, zorder=4)

        # storm marker
        if storm_lon is not None and storm_lat is not None:
            ax.plot(storm_lon, storm_lat, "b*", markersize=14, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.set_xlim(lon_min_ext, lon_max_ext)
        ax.set_ylim(lat_min_ext, lat_max_ext)
        ax.set_title(date_str, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=9)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")

    # shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Storm probability", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    log.info(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--dates",   default="2024-09-25,2024-10-09,2024-10-23",
                        help="Comma-separated dates YYYY-MM-DD")
    parser.add_argument("--lat_min", type=float, default=10.0)
    parser.add_argument("--lat_max", type=float, default=35.0)
    parser.add_argument("--lon_min", type=float, default=-100.0,
                        help="Longitude in [-180,180] convention")
    parser.add_argument("--lon_max", type=float, default=-60.0,
                        help="Longitude in [-180,180] convention")
    # Hurricane Milton landfall location (Cape Canaveral area, ~28.5N 80.5W)
    parser.add_argument("--storm_lat", type=float, default=28.5)
    parser.add_argument("--storm_lon", type=float, default=-80.5,
                        help="Mark on all panels (None to disable)")
    parser.add_argument("--no_storm_marker", action="store_true")
    parser.add_argument("--in_chans", type=int, default=None,
                        help="Override in_chans (4=no elev, 5=with elev). Auto-detected from ckpt if omitted.")
    parser.add_argument("--output",  default="fig_storm_map.png")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")]
    log.info(f"Dates: {dates}")

    # ----------------------------------------------------------------
    # 1. Load coordinates
    # ----------------------------------------------------------------
    log.info("Loading coordinate arrays ...")
    lat, lon, time_arr = load_coords()
    log.info(f"lat: {lat[0]:.2f}..{lat[-1]:.2f} ({len(lat)}), "
             f"lon: {lon[0]:.2f}..{lon[-1]:.2f} ({len(lon)}), "
             f"time: {time_arr[0]}..{time_arr[-1]} ({len(time_arr)})")

    # ----------------------------------------------------------------
    # 2. Load and pad elevation (done once)
    # ----------------------------------------------------------------
    log.info("Loading elevation ...")
    elev_raw = np.load(os.path.join(DATA_DIR, f"elev_{PRECISION}.npy")).astype(np.float32)
    # make_padding works on any shape with lat/lon as last 2 dims
    elev_padded, _ = make_padding(elev_raw, HALF_SIDE)   # (254, 414)
    log.info(f"Elevation padded shape: {elev_padded.shape}")

    # ----------------------------------------------------------------
    # 3. Load model
    # ----------------------------------------------------------------
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log.info(f"Device: {device}")

    log.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # determine in_chans / use_elevation from saved hparams
    # Default: 5 (with elevation). Falls back gracefully.
    in_chans = 5
    use_elevation = True
    try:
        hparams = ckpt.get("hyper_parameters", {})
        cfg_ckpt = hparams.get("cfg", None)
        if cfg_ckpt is not None:
            train_cfg = cfg_ckpt.get("train", {}) if isinstance(cfg_ckpt, dict) else cfg_ckpt.train
            # OmegaConf DictConfig or plain dict
            try:
                use_elevation = bool(train_cfg.get("use_elevation", True))
                in_chans = int(train_cfg.get("in_chans", 5 if use_elevation else 4))
            except Exception:
                use_elevation = bool(getattr(train_cfg, "use_elevation", True))
                in_chans = int(getattr(train_cfg, "in_chans", 5 if use_elevation else 4))
    except Exception as e:
        log.warning(f"Could not read hparams ({e}), defaulting to in_chans={in_chans}")

    # override if explicitly passed
    if args.in_chans is not None:
        in_chans = args.in_chans
        use_elevation = (in_chans == 5)

    log.info(f"in_chans: {in_chans}  use_elevation: {use_elevation}")

    model = GhostWindNet27(in_chans=in_chans)

    # strip Lightning wrapper keys and pos_weight
    state = ckpt["state_dict"]
    cleaned = {}
    for k, v in state.items():
        if k.startswith("net."):
            cleaned[k[4:]] = v   # strip "net."
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")

    model.to(device).eval()
    log.info("Model loaded.")

    # ----------------------------------------------------------------
    # 4. Determine region cells
    # ----------------------------------------------------------------
    cells = region_cells(lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    log.info(f"Region cells: {len(cells)} "
             f"(lat {args.lat_min}-{args.lat_max}, "
             f"lon {args.lon_min}-{args.lon_max})")

    storm_lon = None if args.no_storm_marker else args.storm_lon
    storm_lat = None if args.no_storm_marker else args.storm_lat

    # ----------------------------------------------------------------
    # 5. Run inference for each date
    # ----------------------------------------------------------------
    grids_dates = []
    for date_str in dates:
        log.info(f"--- Date: {date_str} ---")
        t_idx = date_to_idx(date_str, time_arr)

        # check time bounds
        if t_idx - TIME_WIN // 2 < 0 or t_idx + TIME_WIN // 2 >= len(time_arr):
            log.error(f"Date {date_str} too close to data boundary, skipping")
            continue

        log.info("Loading and padding CMIP6 slice ...")
        combined_padded, _ = build_padded_slice(t_idx, elev_padded, use_elevation)
        log.info(f"Combined padded shape: {combined_padded.shape}")

        date_np64 = time_arr[t_idx]
        results = run_inference(model, device, combined_padded, cells, date_np64, in_chans)

        grid, lat_sub, lon_sub_180 = results_to_grid(
            results, lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )
        valid = grid[~np.isnan(grid)]
        log.info(f"Grid shape: {grid.shape}, cells with data: {len(valid)}, "
                 f"prob range: [{valid.min():.3f}, {valid.max():.3f}]" if len(valid) else
                 f"Grid shape: {grid.shape}, NO valid cells")

        grids_dates.append((grid, lat_sub, lon_sub_180, date_str, storm_lon, storm_lat))

    # ----------------------------------------------------------------
    # 6. Plot
    # ----------------------------------------------------------------
    if grids_dates:
        plot_maps(grids_dates, args.output,
                  title="GhostWindNet27 — Storm Probability (Hurricane Milton, Oct 2024)")
    else:
        log.error("No valid results to plot.")


if __name__ == "__main__":
    main()

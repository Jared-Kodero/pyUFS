from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fv3_state import state
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath


def random_colors(n, seed=42):
    """Return n visually distinct random colors.

    Hues are spaced by the golden ratio from a random offset, so repeated
    draws stay separated instead of clustering. The seed keeps the two
    figures consistent with each other and reproducible between runs.
    """
    rng = np.random.default_rng(seed)
    offset = rng.random()
    hues = (offset + np.arange(n) * 0.618033988749895) % 1.0
    sats = rng.uniform(0.55, 0.95, n)
    vals = rng.uniform(0.50, 0.85, n)
    return [to_hex(hsv_to_rgb([h, s, v])) for h, s, v in zip(hues, sats, vals)]


def tile_number(path):
    """Tile index from a grid file name, used to sort tile10 after tile9."""
    stem = Path(path).name.split(".tile")[-1]
    return int(stem.split(".")[0])


def plot_tiles(grid_dir: Path, target_lon: float, target_lat: float):
    grid_dir = Path(grid_dir)
    grid_files = sorted(grid_dir.glob("C*_grid.tile*.nc"), key=tile_number)
    datasets = [xr.open_dataset(f) for f in grid_files if f.exists()]

    # -------------------- Utilities --------------------

    def _to_deg(a):
        a = np.asarray(a)
        if np.nanmax(np.abs(a)) <= (2 * np.pi + 1e-6):
            return np.rad2deg(a)
        return a

    def load_lonlat(ds):
        lon = _to_deg(ds["x"].values)
        lat = _to_deg(ds["y"].values)
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        return lon % 360, lat

    def get_tile_path(ds):
        """Creates a precise Path object following the boundary of the tile."""
        lon, lat = load_lonlat(ds)
        # Trace the perimeter: Bottom -> Right -> Top (rev) -> Left (rev)
        b_lon = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
        b_lat = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
        return MplPath(np.column_stack([b_lon, b_lat]))

    def ortho_project(lon_deg, lat_deg, lon0_deg=0.0, lat0_deg=0.0):
        lon0_deg = lon0_deg % 360
        lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
        lon0, lat0 = np.deg2rad(lon0_deg), np.deg2rad(lat0_deg)
        cx, cy, cz = (
            np.cos(lat0) * np.cos(lon0),
            np.cos(lat0) * np.sin(lon0),
            np.sin(lat0),
        )
        X, Y, Z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
        visible = (X * cx + Y * cy + Z * cz) > 0.0
        x = np.cos(lat) * np.sin(lon - lon0)
        y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
        return x, y, visible

    def build_segments(lon2d, lat2d):
        ny, nx = lon2d.shape
        segs_int, segs_edge = [], []
        for i in range(ny):
            if 0 < i < ny - 1:
                segs_int.append((lon2d[i, :], lat2d[i, :]))
        for j in range(nx):
            if 0 < j < nx - 1:
                segs_int.append((lon2d[:, j], lat2d[:, j]))
        segs_edge = [
            (lon2d[0, :], lat2d[0, :]),
            (lon2d[-1, :], lat2d[-1, :]),
            (lon2d[:, 0], lat2d[:, 0]),
            (lon2d[:, -1], lat2d[:, -1]),
        ]
        return segs_int, segs_edge

    def project_segment(lon, lat, lon0_deg, lat0_deg, mask_path=None):
        """Project segments, masking out points that fall inside the child Path."""
        x, y, vis = ortho_project(lon, lat, lon0_deg, lat0_deg)

        if mask_path is not None:
            # Check which points of the parent line are inside the child polygon
            points = np.column_stack([lon, lat])
            in_child = mask_path.contains_points(points)
            vis = vis & ~in_child  # Only keep points NOT in the child

        chunks, current = [], []
        for xi, yi, vi in zip(x, y, vis):
            if vi:
                current.append((xi, yi))
            elif len(current) >= 2:
                chunks.append(np.array(current))
                current = []
        if len(current) >= 2:
            chunks.append(np.array(current))
        return chunks

    # -------------------- Plotting --------------------

    tile_colors = random_colors(len(datasets))
    fig, ax = plt.subplots(figsize=(10, 10))

    # Outer horizon
    theta = np.linspace(0, 2 * np.pi, 720)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.5)

    resolutions = [
        f"Tile {i + 1} ~{state.res_km[i]:.1f} km" for i in range(6 + state.n_nests + 1)
    ]

    for i, (ds, color) in enumerate(zip(datasets, tile_colors), start=1):
        lon, lat = load_lonlat(ds)

        # Get the path of the child to punch a hole in the current parent
        child_path = None
        if i >= 6 and i < len(datasets):
            child_path = get_tile_path(datasets[i])

        subset = 5
        lon_s, lat_s = lon[::subset, ::subset], lat[::subset, ::subset]
        seg_int, seg_edge = build_segments(lon_s, lat_s)

        # Internal lines (with precise mask)
        for L, A in seg_int:
            chunks = project_segment(L, A, target_lon, target_lat, mask_path=child_path)
            ax.add_collection(LineCollection(chunks, lw=0.3, alpha=0.6, colors=color))

        # Edges (No mask, to ensure the boundaries meet perfectly)
        for L, A in seg_edge:
            chunks = project_segment(L, A, target_lon, target_lat)
            ax.add_collection(LineCollection(chunks, lw=1.2, zorder=3, colors=color))

    # Landmask
    try:
        from cartopy.feature import LAND

        for geom in LAND.geometries():
            # LAND yields MultiPolygon as well as Polygon; .exterior exists
            # only on Polygon, and one AttributeError would drop all land.
            for poly in getattr(geom, "geoms", [geom]):
                pts = np.array(poly.exterior.coords[:])
                x, y, v = ortho_project(pts[:, 0], pts[:, 1], target_lon, target_lat)
                if np.any(v):
                    ax.fill(x[v], y[v], alpha=0.08, zorder=0, color="black")
    except Exception:
        pass

    # telescoping nests with increasing resolution

    # add pathces for legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, color=tile_colors[i + 5], label=c_res)
        for i, c_res in enumerate(resolutions)
    ]
    ax.legend(
        handles=legend_elements, loc="lower left", fontsize="small", framealpha=0.5
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(state.run_dir / "grids.png", dpi=1200, bbox_inches="tight")
    plt.show()

    for ds in datasets:
        ds.close()

    return resolutions  # ["Tile 6 ~x km", "Tile 7 ~y km", ...]


def plot_platecarree(
    lon_min: list, lon_max: list, lat_min: list, lat_max: list, resolutions: list
):
    """
    Plot rectangular geographic domains on a Lambert Conformal projection.

    Parameters
    ----------
    lon_min : list of float
        Western longitude bounds (degrees east).
    lon_max : list of float
        Eastern longitude bounds (degrees east).
    lat_min : list of float
        Southern latitude bounds (degrees north).
    lat_max : list of float
        Northern latitude bounds (degrees north).
    resolutions : list of str, optional
        Legend labels for the nests, as returned by plot_tiles.
    """

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    n_nests = len(lon_min)

    # Same seed as plot_tiles, so nest i keeps the colour of tile 7 + i.
    tile_colors = random_colors(6 + n_nests)[6:]

    for i in range(n_nests):
        width = lon_max[i] - lon_min[i]
        height = lat_max[i] - lat_min[i]

        rect = Rectangle(
            (lon_min[i], lat_min[i]),
            width,
            height,
            linewidth=2,
            edgecolor=tile_colors[i],
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.add_patch(rect)

    ax.set_extent(
        [min(lon_min) - 10, max(lon_max) + 10, min(lat_min) - 10, max(lat_max) + 10],
        crs=ccrs.PlateCarree(),
    )

    legend_elements = [
        Rectangle((0, 0), 1, 1, color=c, label=c_res)
        for c, c_res in zip(tile_colors, resolutions)
    ]
    ax.legend(
        handles=legend_elements, loc="lower left", fontsize="small", framealpha=0.5
    )
    plt.tight_layout()
    plt.savefig(state.run_dir / "nest_grid.png", dpi=1200, bbox_inches="tight")
    plt.show()


def plot_grid():
    import cartopy

    cartopy_data = state.fix_src / "carto"
    cartopy.config["pre_existing_data_dir"] = cartopy_data
    cartopy.config["data_dir"] = cartopy_data

    try:
        resolutions = plot_tiles(state.grid, state.target_lon, state.target_lat)

        if state.n_nests > 0:
            plot_platecarree(
                state.lon_min,
                state.lon_max,
                state.lat_min,
                state.lat_max,
                resolutions[1:],  # drop Tile 6, the parent, keep the nests
            )

    except Exception:
        return

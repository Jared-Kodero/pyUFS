"""Visualisation of FV3 cubed-sphere tiles and their nests.

Figures produced
----------------
grid_faces.png  one panel per global cubed-sphere face, each centred on its own
                centroid, with every nest hosted by that face drawn on it and
                the host mesh masked underneath
nest_grid.png   nest domains on a PlateCarree map

Nest hosting is taken from state.parent_tile, the direct parent tile of each
nest: parent_tile[k] is the tile hosting nest k + 1, which occupies tile 7 + k.
A chain such as [6, 7, 8] telescopes, so all three nests resolve to face 6.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fv3_state import state
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb, to_rgba
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

N_GLOBAL_TILES = 6

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

# Muted mineral and vegetation tones. Ordered so that adjacent nests differ in
# hue and in lightness, which keeps them separable in greyscale printing.
EARTH_TONES = [
    "#B0542F",  # rust
    "#4F6D4E",  # moss
    "#C2913A",  # ochre
    "#3E6B7C",  # slate blue
    "#7A4B2A",  # umber
    "#8A9A5B",  # sage
    "#2F6D5E",  # deep teal
    "#9C6B4F",  # clay
    "#5C6B4A",  # olive drab
    "#A8763E",  # bronze
    "#6E5A46",  # taupe
    "#C2A878",  # sand
]

GLOBAL_TILE_COLOR = "#57544E"  # neutral stone grey for the six global faces
LAND_FILL = "#6E6A60"
HORIZON_COLOR = "#3A3A36"
LAND_FACE = "#EDE6D8"
OCEAN_FACE = "#DCE5E9"
COAST_COLOR = "#66655D"
BORDER_COLOR = "#8C8A82"
GRIDLINE_COLOR = "#B9B4A8"
TEXT_COLOR = "#2B2B28"

# Mesh decimation is set by physical grid spacing, not by array size. Drawn
# lines are held to a roughly constant separation on the page, so a 3 km nest
# covering a small part of a panel is thinned far more than the 100 km face
# around it, and neither saturates to solid colour.
MESH_PAGE_SPACING_IN = 0.10  # target separation between drawn lines, inches
MESH_MIN_LINES = 6  # floor, so a small nest still shows internal structure
MESH_MAX_LINES = 60  # ceiling, bounds cost on the largest supergrids

# IUGG mean radius R1 of the GRS80 ellipsoid (Moritz, 2000).
EARTH_RADIUS_KM = 6371.0088

PANEL_FIG_IN = 3.4  # width of one face panel, inches
FIG_DPI = 1200  # dots per inch for the figure, so a 3.4" panel is 4080 pixels wide


def earth_colors(n, seed=42):
    """Return n earth-tone colours.

    The first len(EARTH_TONES) colours are the fixed palette above, taken in
    order, so a given nest keeps its colour between runs and between figures.
    Beyond that the palette is reused with progressively lower value and higher
    saturation rather than repeated identically. The seed is retained for call
    compatibility and only offsets the starting entry.
    """
    n = max(int(n), 0)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(EARTH_TONES))) if seed is None else 0
    colors = []
    for k in range(n):
        idx = (start + k) % len(EARTH_TONES)
        cycle = k // len(EARTH_TONES)
        rgb = to_rgb(EARTH_TONES[idx])
        if cycle:
            h, s, v = rgb_to_hsv(rgb)
            v = float(np.clip(v * (1.0 - 0.18 * cycle), 0.25, 0.90))
            s = float(np.clip(s * (1.0 + 0.12 * cycle), 0.20, 0.95))
            rgb = hsv_to_rgb([h, s, v])
        colors.append(to_hex(rgb))
    return colors


def random_colors(n, seed=42):
    """Backwards-compatible alias for earth_colors."""
    return earth_colors(n, seed=seed)


def tile_palette(n_nests, seed=42):
    """Return colours for the six global faces followed by n_nests nests.

    The global faces share one neutral colour so the nests are the only
    chromatic features. Element i corresponds to tile i + 1, so nest k
    (tile 6 + k) is at index 5 + k.
    """
    n_nests = max(int(n_nests), 0)
    return [GLOBAL_TILE_COLOR] * N_GLOBAL_TILES + earth_colors(n_nests, seed=seed)


def tile_number(path):
    """Tile index from a grid file name, used to sort tile10 after tile9."""
    stem = Path(path).name.split(".tile")[-1]
    return int(stem.split(".")[0])


# --------------------------------------------------------------------------
# Nest hierarchy
# --------------------------------------------------------------------------


def nest_parents(n_nests):
    """Direct parent tile of each nest, as a list of length n_nests.

    Read from state.parent_tile. Element k is the tile hosting nest k + 1,
    which itself occupies tile 7 + k. If state.parent_tile is absent or of the
    wrong length, a telescoping chain [6, 7, 8, ...] is assumed and used.
    """
    n_nests = max(int(n_nests), 0)
    parents = state.parent_tile
    if not parents:
        parents = []

    if isinstance(parents, int):
        parents = [parents]

    if len(parents) != n_nests:
        parents = [N_GLOBAL_TILES + k for k in range(n_nests)]
    return [int(p) for p in parents]


def nest_tile(k):
    """Tile number of nest k, zero based: nest 0 is tile 7."""
    return N_GLOBAL_TILES + 1 + int(k)


def direct_children(tile, parents):
    """Indices of the nests whose direct parent is the given tile."""
    return [k for k, p in enumerate(parents) if p == int(tile)]


def host_face(k, parents):
    """Global face that ultimately hosts nest k, following the parent chain.

    Returns None if the chain is malformed, for example a cycle or a parent
    tile with no corresponding nest.
    """
    tile = parents[k]
    seen = set()
    while tile > N_GLOBAL_TILES:
        idx = tile - N_GLOBAL_TILES - 1  # tile 7 is nest index 0
        if idx in seen or idx < 0 or idx >= len(parents):
            return None
        seen.add(idx)
        tile = parents[idx]
    return tile if 1 <= tile <= N_GLOBAL_TILES else None


def face_nests(face, parents):
    """Nest indices hosted by a global face, directly or through a chain.

    Returned in ascending tile order, which is coarse to fine for a telescope,
    so drawing in this order puts each child on top of its parent.
    """
    return [k for k in range(len(parents)) if host_face(k, parents) == int(face)]


# --------------------------------------------------------------------------
# Geometry utilities
# --------------------------------------------------------------------------


def _to_deg(a):
    a = np.asarray(a)
    if np.nanmax(np.abs(a)) <= (2 * np.pi + 1e-6):
        return np.rad2deg(a)
    return a


def load_lonlat(ds):
    """Return (lon, lat) in degrees as 2-D arrays, lon wrapped to [0, 360)."""
    lon = _to_deg(ds["x"].values)
    lat = _to_deg(ds["y"].values)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return lon % 360, lat


def get_tile_path(ds):
    """Path object following the boundary of a tile in lon-lat space."""
    lon, lat = load_lonlat(ds)
    b_lon = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
    b_lat = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
    return MplPath(np.column_stack([b_lon, b_lat]))


def ortho_project(lon_deg, lat_deg, lon0_deg=0.0, lat0_deg=0.0, roll_deg=0.0):
    """Orthographic projection onto the tangent plane at ``(lon0, lat0)``.

    ``roll_deg`` rotates the tangent plane clockwise.  Setting it to the angle
    of the tile's increasing-i direction preserves the native cubed-sphere
    orientation instead of forcing geographic north to the top.
    """
    lon0_deg = lon0_deg % 360
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    lon0, lat0 = np.deg2rad(lon0_deg), np.deg2rad(lat0_deg)
    cx, cy, cz = np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)
    X, Y, Z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    visible = (X * cx + Y * cy + Z * cz) > 0.0
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)

    if roll_deg:
        angle = np.deg2rad(roll_deg)
        c, s = np.cos(angle), np.sin(angle)
        x, y = c * x + s * y, -s * x + c * y

    return x, y, visible


def computational_roll_deg(lon, lat, lon0, lat0):
    """Clockwise roll that makes the tile's increasing-i axis horizontal."""
    j = lon.shape[0] // 2
    i = lon.shape[1] // 2
    i0 = max(i - 1, 0)
    i1 = min(i + 1, lon.shape[1] - 1)
    x, y, _ = ortho_project(
        lon[j, [i0, i1]], lat[j, [i0, i1]], lon0, lat0, roll_deg=0.0
    )
    return float(np.rad2deg(np.arctan2(y[1] - y[0], x[1] - x[0])))


def projected_tile_path(lon, lat, lon0, lat0, roll_deg):
    """Closed tile boundary as a path in the rolled tangent plane."""
    b_lon = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
    b_lat = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
    x, y, visible = ortho_project(b_lon, b_lat, lon0, lat0, roll_deg)
    vertices = np.column_stack([x[visible], y[visible]])
    return MplPath(vertices, closed=True)


def tile_center(lon, lat):
    """Centroid of a tile as (lon, lat) in degrees.

    The mean is taken over unit vectors rather than over angles, so the result
    is well defined across the dateline and at the poles.
    """
    lonr, latr = np.deg2rad(lon), np.deg2rad(lat)
    x = np.nanmean(np.cos(latr) * np.cos(lonr))
    y = np.nanmean(np.cos(latr) * np.sin(lonr))
    z = np.nanmean(np.sin(latr))
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-12:
        return 0.0, 0.0
    return float(np.rad2deg(np.arctan2(y, x)) % 360), float(
        np.rad2deg(np.arcsin(z / r))
    )


def great_circle_km(lon1, lat1, lon2, lat2, radius=EARTH_RADIUS_KM):
    """Haversine distance between two points in degrees, returned in km."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def grid_spacing_km(lon, lat):
    """Spacing between adjacent grid points near the tile centre, in km.

    Returned as (dy, dx). Measured from the file rather than taken from
    state.res_km, so it is correct whether the file is a supergrid (spacing
    res_km / 2) or a cell-centre grid, and it needs no assumption about the
    ordering of res_km. Local isotropy of the gnomonic cubed sphere makes a
    single central sample adequate for setting line density.
    """
    ny, nx = lon.shape
    j, i = ny // 2, nx // 2
    dy = great_circle_km(lon[j, i], lat[j, i], lon[j + 1, i], lat[j + 1, i])
    dx = great_circle_km(lon[j, i], lat[j, i], lon[j, i + 1], lat[j, i + 1])
    return float(max(dy, 1e-6)), float(max(dx, 1e-6))


def page_km_per_inch(span_units, width_inch, radius=EARTH_RADIUS_KM):
    """Map scale of an orthographic axis, in km per inch of page.

    The projection returns coordinates in Earth radii, so an axis spanning
    span_units across width_inch has scale radius * span_units / width_inch.
    """
    return radius * float(span_units) / max(float(width_inch), 1e-6)


def mesh_stride(
    n,
    spacing_km,
    km_per_inch,
    page_spacing=MESH_PAGE_SPACING_IN,
    min_lines=MESH_MIN_LINES,
    max_lines=MESH_MAX_LINES,
):
    """Stride that holds drawn lines near page_spacing inches apart.

    A separation of page_spacing inches corresponds to a distance
    d = page_spacing * km_per_inch on the sphere, so the stride is
    s = ceil(d / spacing_km) for a tile of grid spacing spacing_km. The stride
    is then bounded so that between min_lines and max_lines are drawn whatever
    the size of the tile.
    """
    n = int(n)
    target_km = float(page_spacing) * float(km_per_inch)
    s = int(np.ceil(target_km / float(spacing_km)))
    s = max(s, int(np.ceil(n / float(max_lines))))  # never exceed max_lines
    s = min(s, max(1, n // int(min_lines)))  # never fall below min_lines
    return max(s, 1)


def _stride_index(n, stride):
    """Strided indices that always retain the last point, so the tile edge is
    the true edge rather than the last multiple of the stride."""
    idx = np.arange(0, n, max(int(stride), 1))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def subset_mesh(lon, lat, km_per_inch, page_spacing=MESH_PAGE_SPACING_IN):
    """Decimate a mesh to a constant on-page line density.

    Parameters
    ----------
    lon, lat : ndarray
        Tile coordinates in degrees.
    km_per_inch : float
        Map scale of the target axis, from page_km_per_inch.
    page_spacing : float
        Target separation between drawn lines, inches.
    """
    ny, nx = lon.shape
    dy_km, dx_km = grid_spacing_km(lon, lat)
    sy = mesh_stride(ny, dy_km, km_per_inch, page_spacing)
    sx = mesh_stride(nx, dx_km, km_per_inch, page_spacing)
    sel = np.ix_(_stride_index(ny, sy), _stride_index(nx, sx))
    return lon[sel], lat[sel]


def build_segments(lon2d, lat2d):
    """Split a mesh into interior lines and the four boundary lines."""
    ny, nx = lon2d.shape
    segs_int = []
    for i in range(1, ny - 1):
        segs_int.append((lon2d[i, :], lat2d[i, :]))
    for j in range(1, nx - 1):
        segs_int.append((lon2d[:, j], lat2d[:, j]))
    segs_edge = [
        (lon2d[0, :], lat2d[0, :]),
        (lon2d[-1, :], lat2d[-1, :]),
        (lon2d[:, 0], lat2d[:, 0]),
        (lon2d[:, -1], lat2d[:, -1]),
    ]
    return segs_int, segs_edge


def _inside_any(paths, lon, lat):
    """Mask of points falling inside any of the given lon-lat paths."""
    pts = np.column_stack([lon, lat])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for p in paths:
        inside |= p.contains_points(pts)
    return inside


def project_segment(lon, lat, lon0_deg, lat0_deg, mask_paths=None, roll_deg=0.0):
    """Project one grid line, dropping points hidden by the limb or falling
    inside any mask path, and split the remainder into continuous chunks."""
    x, y, vis = ortho_project(lon, lat, lon0_deg, lat0_deg, roll_deg)

    if mask_paths:
        vis = vis & ~_inside_any(mask_paths, lon, lat)

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


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------


def draw_tile_mesh(
    ax,
    lon,
    lat,
    color,
    lon0,
    lat0,
    km_per_inch,
    mask_paths=None,
    roll_deg=0.0,
    page_spacing=MESH_PAGE_SPACING_IN,
    interior_lw=0.3,
    edge_lw=1.2,
    alpha=0.6,
    zorder=2,
):
    """Draw one tile mesh in orthographic view centred on (lon0, lat0).

    km_per_inch is the map scale of the target axis and sets the decimation, so
    the mesh reads at the same density whether the tile is a 100 km face
    filling the panel or a 3 km nest covering a few percent of it. Interior
    lines falling inside mask_paths are dropped, which is how a host face is
    hollowed out under the nests it carries.
    """
    lon_s, lat_s = subset_mesh(lon, lat, km_per_inch, page_spacing)
    seg_int, seg_edge = build_segments(lon_s, lat_s)

    for L, A in seg_int:
        chunks = project_segment(
            L, A, lon0, lat0, mask_paths=mask_paths, roll_deg=roll_deg
        )
        if chunks:
            ax.add_collection(
                LineCollection(
                    chunks, lw=interior_lw, alpha=alpha, colors=color, zorder=zorder
                )
            )

    # Edges are never masked, so the boundary of a host and the boundary of its
    # nest remain coincident.
    for L, A in seg_edge:
        chunks = project_segment(L, A, lon0, lat0, roll_deg=roll_deg)
        if chunks:
            ax.add_collection(
                LineCollection(chunks, lw=edge_lw, colors=color, zorder=zorder + 1)
            )


def draw_land(
    ax,
    lon0,
    lat0,
    roll_deg=0.0,
    clip_path=None,
    color=LAND_FILL,
    alpha=0.3,
    zorder=0,
):
    """Fill land only inside the current tile boundary."""
    try:
        for geom in cfeature.LAND.geometries():
            for poly in getattr(geom, "geoms", [geom]):
                pts = np.asarray(poly.exterior.coords[:])
                x, y, v = ortho_project(
                    pts[:, 0], pts[:, 1], lon0, lat0, roll_deg=roll_deg
                )
                if np.any(v):
                    patches = ax.fill(
                        x[v], y[v], alpha=alpha, zorder=zorder, color=color
                    )
                    if clip_path is not None:
                        for patch in patches:
                            patch.set_clip_path(clip_path)
    except Exception:
        return


def draw_horizon(
    ax, roll_deg=0.0, clip_path=None, lw=0.8, color=HORIZON_COLOR, zorder=1
):
    theta = np.linspace(0, 2 * np.pi, 720)
    (line,) = ax.plot(np.cos(theta), np.sin(theta), color=color, lw=lw, zorder=zorder)
    if clip_path is not None:
        line.set_clip_path(clip_path)


def tile_extent(lon, lat, lon0, lat0, roll_deg=0.0, pad=0.08):
    """Axis limits enclosing the visible part of a tile, with fractional pad."""
    x, y, v = ortho_project(lon, lat, lon0, lat0, roll_deg)
    if not np.any(v):
        return (-1.05, 1.05), (-1.05, 1.05)
    x, y = x[v], y[v]
    dx = max(x.max() - x.min(), 1e-3)
    dy = max(y.max() - y.min(), 1e-3)
    half = 0.5 * max(dx, dy) * (1.0 + pad)
    cx, cy = 0.5 * (x.max() + x.min()), 0.5 * (y.max() + y.min())
    return (cx - half, cx + half), (cy - half, cy + half)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def tile_labels(n_tiles, parents):
    """Panel titles for the six faces and legend labels for the nests.

    res_km[0] is the global resolution and res_km[k] the resolution of nest k,
    which occupies tile 6 + k.
    """
    res = list(getattr(state, "res_km", []))

    def resolved(tile):
        idx = 0 if tile <= N_GLOBAL_TILES else tile - N_GLOBAL_TILES
        return f"Tile {tile} ~{res[idx]:.1f} km" if idx < len(res) else f"Tile {tile}"

    face_titles = [resolved(t) for t in range(1, N_GLOBAL_TILES + 1)]
    nest_labels = [
        f"{resolved(nest_tile(k))} on tile {parents[k]}" for k in range(len(parents))
    ]
    return face_titles, nest_labels


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_faces(datasets, tile_colors, face_titles, nest_labels, parents, out_path):
    """One panel per global face, with the nests it hosts drawn on it.

    Each panel is an orthographic view centred on the centroid of its own face
    and zoomed to it, so all six faces are legible in one figure. A nest is
    placed on the face returned by host_face, which follows parent_tile up the
    chain, so a telescope rooted at tile 6 appears entirely on the tile 6 panel.
    Every tile is hollowed out under its direct children, leaving each mesh
    visible only where it is the finest grid present.
    """
    n_faces = min(N_GLOBAL_TILES, len(datasets))
    ncols = 3
    nrows = int(np.ceil(n_faces / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PANEL_FIG_IN * ncols, (PANEL_FIG_IN + 0.2) * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    def paths_of_children(tile):
        paths = []
        for k in direct_children(tile, parents):
            idx = nest_tile(k) - 1
            if idx < len(datasets):
                paths.append(get_tile_path(datasets[idx]))
        return paths

    for f in range(1, n_faces + 1):
        ax = axes[f - 1]
        lon, lat = load_lonlat(datasets[f - 1])
        lon0, lat0 = tile_center(lon, lat)
        roll_deg = 0.0
        # roll_deg = computational_roll_deg(lon, lat, lon0, lat0)

        # Limits are fixed first: each panel is zoomed to its own face, so its
        # map scale, and therefore its stride, follows from the zoom.
        xlim, ylim = tile_extent(lon, lat, lon0, lat0, roll_deg=roll_deg)
        km_per_inch = page_km_per_inch(xlim[1] - xlim[0], PANEL_FIG_IN)

        tile_clip = PathPatch(
            projected_tile_path(lon, lat, lon0, lat0, roll_deg),
            transform=ax.transData,
            facecolor="white",
            edgecolor="none",
            zorder=-1,
        )
        ax.add_patch(tile_clip)
        draw_land(ax, lon0, lat0, roll_deg=roll_deg, clip_path=tile_clip)
        draw_horizon(ax, roll_deg=roll_deg, clip_path=tile_clip)

        draw_tile_mesh(
            ax,
            lon,
            lat,
            tile_colors[f - 1],
            lon0,
            lat0,
            km_per_inch,
            mask_paths=paths_of_children(f),
            roll_deg=roll_deg,
            interior_lw=0.25,
            edge_lw=1.0,
            alpha=0.55,
            zorder=2,
        )

        hosted = face_nests(f, parents)
        for rank, k in enumerate(hosted):
            idx = nest_tile(k) - 1
            if idx >= len(datasets):
                continue
            nlon, nlat = load_lonlat(datasets[idx])
            draw_tile_mesh(
                ax,
                nlon,
                nlat,
                tile_colors[idx],
                lon0,
                lat0,
                km_per_inch,
                mask_paths=paths_of_children(nest_tile(k)),
                roll_deg=roll_deg,
                interior_lw=0.3,
                edge_lw=1.1,
                alpha=0.85,
                zorder=4 + 2 * rank,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        title = face_titles[f - 1]
        if hosted:
            title += "\n" + ", ".join(f"T{nest_tile(k)}" for k in hosted)
        ax.set_title(title, fontsize=9, color=TEXT_COLOR)
        ax.axis("off")

    for ax in axes[n_faces:]:
        ax.axis("off")

    handles = [
        Rectangle((0, 0), 1, 1, color=tile_colors[nest_tile(k) - 1], label=lab)
        for k, lab in enumerate(nest_labels)
        if (nest_tile(k) - 1) < len(tile_colors)
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(handles), 3),
            fontsize="small",
            frameon=False,
        )

    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_nests(
    lon_min: list, lon_max: list, lat_min: list, lat_max: list, resolutions: list
):
    """Plot nest domains on a projection selected from their union bounds.

    Candidate set is restricted to Robinson, cylindrical equidistant
    (Plate Carree), Lambert conformal conic, and polar stereographic.
    Assignment follows the WPS/ARW convention: polar stereographic for
    high-latitude domains, Lambert conformal conic for mid-latitudes,
    cylindrical equidistant for low-latitude and equator-straddling
    domains, and Robinson for whole-world spans (Robinson, 1974). Conic
    standard parallels use the one-sixth rule of Deetz and Adams (1934).

    Longitudes are assumed to lie in [-180, 180] degrees east with
    lon_min < lon_max for every nest, so no domain crosses the
    antimeridian.

    Parameters
    ----------
    lon_min, lon_max : list of float
        Western and eastern longitude bounds, degrees east.
    lat_min, lat_max : list of float
        Southern and northern latitude bounds, degrees north.
    resolutions : list of str
        Legend labels for the nests, as returned by plot_tiles.
    """
    n_nests = len(lon_min)
    if n_nests == 0:
        return

    tile_colors = tile_palette(n_nests)[N_GLOBAL_TILES:]

    # Union bounding box over all nests.
    lon_w, lon_e = float(min(lon_min)), float(max(lon_max))
    lat_s, lat_n = float(min(lat_min)), float(max(lat_max))
    lon_c, lat_c = 0.5 * (lon_w + lon_e), 0.5 * (lat_s + lat_n)
    d_lon, d_lat = lon_e - lon_w, lat_n - lat_s

    # Projection selection. Order matters: a conic cannot contain a pole,
    # and its cone constant collapses across the equator, so both cases are
    # tested before the Lambert branch.
    if d_lon >= 300.0 and d_lat >= 120.0:
        proj = ccrs.Robinson(central_longitude=lon_c)
    elif d_lon >= 300.0:
        proj = ccrs.PlateCarree(central_longitude=lon_c)
    elif abs(lat_c) >= 70.0 or lat_n >= 85.0 or lat_s <= -85.0:
        true_scale = float(np.clip(abs(lat_c), 60.0, 89.0))
        if lat_c >= 0.0:
            proj = ccrs.NorthPolarStereo(
                central_longitude=lon_c, true_scale_latitude=true_scale
            )
        else:
            proj = ccrs.SouthPolarStereo(
                central_longitude=lon_c, true_scale_latitude=-true_scale
            )

    elif abs(lat_c) <= 25.0 or lat_s * lat_n < 0.0:
        proj = ccrs.PlateCarree(central_longitude=lon_c)

    else:
        # One-sixth rule; collapses to the tangent case for a thin domain.
        if d_lat < 1.0:
            parallels = (lat_c,)
        else:
            parallels = (lat_s + d_lat / 6.0, lat_n - d_lat / 6.0)
        proj = ccrs.LambertConformal(
            central_longitude=lon_c,
            central_latitude=lat_c,
            standard_parallels=parallels,
            cutoff=-30.0 if lat_c >= 0.0 else 30.0,
        )

    pad_x = max(0.30 * d_lon, 3.0)
    pad_y = max(0.30 * d_lat, 3.0)
    extent = [
        max(lon_w - pad_x, -180.0),
        min(lon_e + pad_x, 180.0),
        max(lat_s - pad_y, -90.0),
        min(lat_n + pad_y, 90.0),
    ]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_FACE, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_FACE, zorder=0)
    ax.add_feature(
        cfeature.LAKES,
        facecolor=OCEAN_FACE,
        edgecolor=COAST_COLOR,
        linewidth=0.3,
        zorder=1,
    )
    ax.add_feature(cfeature.STATES, edgecolor=BORDER_COLOR, linewidth=0.3, zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COAST_COLOR, linewidth=0.6, zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color=GRIDLINE_COLOR,
        linestyle=":",
        alpha=0.8,
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": TEXT_COLOR}
    gl.ylabel_style = {"size": 8, "color": TEXT_COLOR}
    if hasattr(gl, "rotate_labels"):
        # Meridians converge off-vertical in conic and stereographic frames.
        gl.rotate_labels = False

    # Draw the largest domain first so smaller nests are never hidden.
    areas = [
        (lon_max[i] - lon_min[i]) * (lat_max[i] - lat_min[i]) for i in range(n_nests)
    ]
    order = np.argsort(areas)[::-1]

    for i in order:
        color = tile_colors[i]
        # Edges are densified because a four-vertex rectangle renders as
        # straight lines in a curvilinear frame and would not follow lines
        # of constant latitude or longitude.
        n_edge = 120
        xs = np.linspace(lon_min[i], lon_max[i], n_edge)
        ys = np.linspace(lat_min[i], lat_max[i], n_edge)
        verts = np.column_stack(
            [
                np.concatenate(
                    [
                        xs,
                        np.full(n_edge, lon_max[i]),
                        xs[::-1],
                        np.full(n_edge, lon_min[i]),
                    ]
                ),
                np.concatenate(
                    [
                        np.full(n_edge, lat_min[i]),
                        ys,
                        np.full(n_edge, lat_max[i]),
                        ys[::-1],
                    ]
                ),
            ]
        )
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                linewidth=1.8,
                edgecolor=color,
                facecolor=to_rgba(color, 0.08),
                transform=ccrs.PlateCarree(),
                zorder=3 + int(n_nests - i),
            )
        )
        ax.text(
            lon_min[i],
            lat_max[i],
            f"T{nest_tile(i)}",
            transform=ccrs.PlateCarree(),
            fontsize=7,
            color="white",
            va="bottom",
            ha="left",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="none"),
        )

    legend_elements = [
        Rectangle((0, 0), 1, 1, color=c, label=lab)
        for c, lab in zip(tile_colors, resolutions)
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize="small",
        framealpha=0.85,
        edgecolor="none",
    )

    fig.savefig(state.run_dir / "nest_grids.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_tiles(grid_dir: Path):
    """Draw grid_faces.png: the six global faces with their nests on them.

    Parameters
    ----------
    grid_dir : Path
        Directory holding C*_grid.tile*.nc supergrid files.

    Returns
    -------
    list of str
        Legend labels for the nests, in the form "Tile 7 ~3.0 km on tile 6".
    """
    grid_dir = Path(grid_dir)
    grid_files = sorted(grid_dir.glob("C*_grid.tile*.nc"), key=tile_number)
    datasets = [xr.open_dataset(f) for f in grid_files if f.exists()]
    if not datasets:
        return []

    try:
        n_nests = max(len(datasets) - N_GLOBAL_TILES, 0)
        parents = nest_parents(n_nests)
        tile_colors = tile_palette(n_nests)
        face_titles, nest_labels = tile_labels(len(datasets), parents)

        plot_faces(
            datasets,
            tile_colors,
            face_titles,
            nest_labels,
            parents,
            state.run_dir / "grid_faces.png",
        )
    finally:
        for ds in datasets:
            ds.close()

    return nest_labels


def plot_grid():
    """Entry point. Configures the cartopy cache and produces both figures."""
    import cartopy

    cartopy_data = state.fix_src / "carto"
    cartopy.config["pre_existing_data_dir"] = cartopy_data
    cartopy.config["data_dir"] = cartopy_data

    try:
        nest_labels = plot_tiles(state.grid)

        if state.n_nests > 0:
            plot_nests(
                state.lon_min,
                state.lon_max,
                state.lat_min,
                state.lat_max,
                nest_labels,
            )
    except Exception:
        return

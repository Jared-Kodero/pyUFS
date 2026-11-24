# Keys of the `sst_perturbation` section of the plugins file.
#   delta   K added to open-water SST (required); negative cools.
#   months  calendar months of the climatology that change.
#   lat     [south, north] degrees north.
#   lon     [west, east] degrees east (0 to 360, west > east wraps the date line); null: all.
#   taper   degrees over which the change rises from 0 at the box edge to delta.
DEFAULTS = {
    "delta": None,
    "months": [6, 7, 8],
    "lat": [50.0, 80.0],
    "lon": None,
    "taper": 5.0,
}
HASHED = True

# distance-rs

`distance-rs` is a Rust/PyO3 Python extension for raster distance accumulation and
continuous path extraction. It is intended as a foundation for replacing ArcGIS-style
`DistanceAccumulation` and `OptimalPathAsLine` workflows without reducing the solver
to an 8-neighbor graph shortest-path problem.

The distance solver uses an accepted-front method with Esri-style local Eikonal
updates. Candidate nodes are updated from accepted points and accepted front
segments in the 3-by-3 neighborhood, which lets the front cross cells at
continuous angles without letting updates jump over intervening cost or barrier
cells.

## Features

- Cost surface support.
- Elevation-aware surface distance.
- Vertical relative moving angle (VRMA) factors:
  `binary`, `linear`, `inverse_linear`, `symmetric_linear`,
  `symmetric_inverse_linear`, `cos`, `sec`, `cos_sec`, `sec_cos`,
  `hiking_time`, and `bidir_hiking_time`.
- Barrier masks and NoData handling via non-finite cost/elevation values.
- Continuous path extraction from the parent field produced by accumulation.
- `uv` + `maturin` development setup.

## Development

The repository pins Python 3.13 in `.python-version` so geospatial
dependencies such as Fiona can resolve prebuilt wheels on macOS.
The base package depends only on NumPy; install the `geo` extra for GeoTIFF,
GeoJSON, GeoPackage, and Shapely adapters.

```bash
uv sync --extra geo
uv run maturin develop
uv run pytest
```

## Baseline Benchmark

Run the local raster baseline benchmark with:

```bash
uv run maturin develop --release
uv run python benchmarks/baseline.py --sizes 16 32 --repeats 3
```

The baseline is a pure Python 8-neighbor raster Dijkstra solver using the same
cell-to-cell cost, elevation, vertical-factor, and barrier rules exposed by the
Python API. It reports solution deltas and timing against `distance_rs`. This is
useful for regression testing and for understanding how much the ordered-upwind
surface differs from a conventional neighbor-link raster solution; it is not a
ground truth for anisotropic continuous travel.

Use `--json path/to/results.json` to save machine-readable results, and
`--cases all` to include the one-way downslope binary vertical-factor case.

To include WhiteboxTools CostDistance as a customer-facing comparison:

```bash
uv sync --group whitebox
uv run maturin develop --release
uv run python benchmarks/baseline.py \
  --baselines raster whitebox \
  --cases flat cost \
  --sizes 16 32 \
  --repeats 3 \
  --json results/whitebox-comparison.json
```

Whitebox `CostDistance` supports a source raster and cost/friction raster. The
benchmark therefore runs it for cases that can be represented as isotropic
cost-distance with barriers encoded as NoData in the cost raster. Cases requiring
surface elevation distance or vertical factors are reported as skipped for
Whitebox rather than compared against a different model.

## Five-Mile Route Plot

Generate a customer-facing route visualization over a synthetic five-mile,
one-meter-resolution corridor:

```bash
uv sync --group plot --group whitebox
uv run maturin develop --release
uv run python examples/five_mile_route.py \
  --solvers all \
  --output-dir results/five-mile
```

The default domain is 5 miles long by 101 meters wide at 1 meter resolution.
The script saves a route overlay and accumulation panels over the cost surface.
For a quick smoke test, use smaller dimensions such as
`--length-m 300 --width-m 61`.

## Barrier Detour Plot

Generate a smaller route visualization comparing ordered upwind against the
shared 8-neighbor raster Dijkstra baseline. White cells are impassable barriers,
while the source and destination use non-white markers:

```bash
uv run python examples/barrier_detour.py
```

The script saves `results/barrier-detour/barrier_detour.png`.

## Zig-Zag Comparison

Generate a flat-cost any-angle comparison with 45, 60, 75, and 85 degree
routes. The 45-degree case is exactly representable by 8-neighbor Dijkstra,
while nearby headings force Dijkstra into longer stair-step paths with much
larger cumulative cross-track area between the traced route and the optimal
straight line. The plot also overlays the exact analytic envelope of equal-cost
Dijkstra routes in both the route and cumulative-area panels, with the
stored-parent route highlighted inside that envelope:

```bash
uv run python examples/zig_zag_comparison.py --output-dir results/zig-zag-comparison
```

The script saves `results/zig-zag-comparison/zig_zag_comparison.png`.

## Maze Route Plot

Generate a deterministic maze route visualization comparing ordered upwind
against the shared 8-neighbor raster Dijkstra baseline:

```bash
uv run python examples/maze_route.py
```

The script saves `results/maze-route/maze_route.png`.

## Ridge Pass Showcase

Generate a terrain example designed to make the solver differences visible:
Whitebox `CostDistance` sees only a cheap friction road across a ridge,
8-neighbor Dijkstra receives the same elevation and vertical-factor cutoff as
`distance-rs` but remains grid-constrained, and ordered upwind uses continuous
front updates through the lower saddle:

```bash
uv sync --group plot --group whitebox
uv run --group whitebox python examples/ridge_pass_showcase.py
```

The script saves `results/ridge-pass-showcase/ridge_pass_showcase.png` and a
JSON summary. If Whitebox is not installed, run
`--solvers ordered_upwind raster` to generate the distance-rs/Dijkstra overlay.

## GeoTIFF Adapter

Use `load_surface` when you want a solver-ready raster surface from GIS files,
and `route_path` when you want a multi-leg map-coordinate route:

```python
from distance_rs import (
    CostRaster,
    GeoBarriers,
    GeoPoints,
    GridSpec,
    load_surface,
    route_path,
)

surface = load_surface(
    CostRaster(
        "land_use.tif",
        values={1: 1.0, 2: 1.8, 3: 4.0},
        blocked_values={99},
    ),
    elevation="elevation.tif",
    barriers=GeoBarriers("barriers.gpkg"),
    grid=GridSpec(resolution=1.0),
)

route = route_path(
    CostRaster("cost.tif", values={1: 1.0, 2: 1.8, 3: 4.0}),
    GeoPoints(
        [(76.612, 39.291), (76.601, 39.303), (76.589, 39.317)],
        crs="EPSG:4326",
    ),
    barriers=GeoBarriers("barriers.geojson"),
    elevation="elevation.tif",
    grid=GridSpec(margin=250.0),
    vertical_factor="bidir_hiking_time",
    baseline_speed=5.0,
)

route.path_xy
route.legs[0].metrics
route.metrics
```

Raster paths infer CRS, transform, resolution, and bounds from the file unless
overridden by `GridSpec`. Vector files infer CRS from file metadata; GeoJSON
defaults to `EPSG:4326`. Plain coordinate lists and Shapely geometries must be
wrapped in `GeoPoints(..., crs=...)` or `GeoBarriers(..., crs=...)`. `margin`
crops each route leg to the start/end bounding box plus that many map units,
snapped to the cost raster grid. The native solver uses a local 3-by-3 Eikonal
stencil for ArcGIS-style behavior.

To exercise the cropped GeoTIFF path on large local files, generate synthetic
8000 x 8000 rasters at 1.5 meter resolution, route across a small corridor, and
compare ordered upwind against the simpler 8-neighbor raster Dijkstra baseline:

```bash
uv run --group plot python examples/large_geotiff_route.py
```

The generated GeoTIFFs are written under ignored `data/large-geotiff-route/`,
and route outputs are written under ignored `results/large-geotiff-route/`,
including a terrain/land-use comparison map PNG.
The synthetic route includes irregular closure polygons, wetland/talus/forest
land-use classes, and a vertical-factor cutoff that makes steep ridge faces
impassable.

## Python Example

```python
import numpy as np
from distance_rs import (
    RasterGrid,
    RasterSurface,
    distance_accumulation,
    optimal_path_as_line,
)

cost = np.ones((100, 100), dtype=float)
elevation = np.zeros((100, 100), dtype=float)
barriers = np.zeros((100, 100), dtype=bool)
barriers[20:80, 40] = True

surface = RasterSurface(
    cost,
    grid=RasterGrid(cell_size=1.0),
    elevation=elevation,
    barriers=barriers,
)

result = distance_accumulation(
    surface,
    source=(50, 50),
    vertical_factor={"type": "bidir_hiking_time"},
)

line = optimal_path_as_line(result, destination=(90, 90))
```

The native solver uses a local 3-by-3 Eikonal stencil to avoid skipping
intervening raster costs or barriers.

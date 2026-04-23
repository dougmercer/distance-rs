# distance-rs

`distance-rs` is a Rust/PyO3 Python extension for raster distance accumulation and
continuous path extraction. It is intended as a foundation for replacing ArcGIS-style
`DistanceAccumulation` and `OptimalPathAsLine` workflows without reducing the solver
to an 8-neighbor graph shortest-path problem.

The distance solver uses an Ordered Upwind style accepted-front method. Candidate
nodes are updated from accepted points and accepted front segments inside a
configurable continuous stencil. Segment updates minimize over interpolated points
on the accepted front, which lets the front cross cells at continuous angles instead
of only along graph edges.

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

Use `geo_distance_accumulation` to align cost/elevation GeoTIFFs to one raster
grid, project GeoJSON/GeoPackage/Shapely LineString waypoints into that grid,
and trace the result back in map coordinates:

```python
from distance_rs import geo_distance_accumulation, geo_optimal_path_as_line

result = geo_distance_accumulation(
    "land_use.tif",
    elevation_path="elevation.tif",
    waypoints="route.gpkg",
    land_use_costs={1: 1.0, 2: 1.8, 3: 4.0},
    barrier_values={99},
    crop_buffer=250.0,
    stencil_radius=80.0,
)

line_map_xy = geo_optimal_path_as_line(result, reverse=True)
```

When `crop_buffer` and at least two waypoints are provided, the adapter only
reads/reprojects the target grid cells inside the start/end bounding box
buffered by that radius. `stencil_radius` controls the Ordered Upwind solver
stencil in map units.

For full routes through multiple waypoints, `compute_optimal_path` runs each
consecutive leg, stitches the map-coordinate polylines, and returns leg and total
metrics:

```python
from distance_rs import compute_optimal_path

route = compute_optimal_path(
    "cost.tif",
    [(76.612, 39.291), (76.601, 39.303), (76.589, 39.317)],
    barriers="barriers.geojson",
    elevation="elevation.tif",
    vertical_factor="bidir_hiking_time",
    crop_buffer=250.0,
    stencil_radius=80.0,
    baseline_speed=5.0,
    waypoint_crs="EPSG:4326",
)

route.path_xy
route.legs[0].metrics
route.metrics
```

Plain coordinate lists passed to geospatial helpers require `waypoint_crs`; use
`EPSG:4326` for lon/lat. Cost raster values are treated as slowdown factors:
`1.0` means baseline speed and `2.0` means twice the travel time. Most vertical
factors are dimensionless multipliers; the hiking-time factors already produce
hours.

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
from distance_rs import distance_accumulation, optimal_path_as_line

sources = np.zeros((100, 100), dtype=bool)
sources[50, 50] = True

cost = np.ones((100, 100), dtype=float)
elevation = np.zeros((100, 100), dtype=float)
barriers = np.zeros((100, 100), dtype=bool)
barriers[20:80, 40] = True

result = distance_accumulation(
    sources,
    cost_surface=cost,
    elevation=elevation,
    vertical_factor={"type": "bidir_hiking_time"},
    barriers=barriers,
    cell_size=1.0,
    search_radius=6.0,
)

line = optimal_path_as_line(result, destination=(90, 90))
```

The `search_radius` controls the Ordered Upwind stencil in map units. Larger values
increase the accepted-front region considered for anisotropic updates and cost more
CPU time.

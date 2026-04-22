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

```bash
uv sync
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

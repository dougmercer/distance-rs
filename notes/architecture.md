---
marp: true
title: Dismounted Mobility Solver Options
paginate: true
---

# Next Steps for a Mobility Solver

Requirements, existing tool gaps, deployment tradeoffs, and the case for a Rust solver.

---

# Executive Takeaway

Most off-the-shelf options are not sufficient:

- cost-only tools miss slope behavior
- direction-independent tools cannot model vertical factors
- graph-based tools introduce route geometry artifacts
- ArcGIS provides good movement model but poor cloud deployment story (vendor lock in, ESRI-specific deployment patterns)
- cloud-friendly tools still need the right movement model

**Recommendation:** create our own Rust solver with Python bindings and GIS adapters around it that supports our desired movement model AND is easy to deploy to a server.

---

# Required Inputs

| Input | Needed Because |
| --- | --- |
| Friction / cost raster | Land cover, vegetation, roads, soil, water; etc. |
| Elevation raster | Slope, surface distance; terrain-aware movement |
| Vertical factor model | Defines uphill / downhill penalties and cutoffs (i.e., too steep) |
| Barriers | Blocks impassable or prohibited areas |
| Sources / destinations | Supports route endpoints, multiple sources, or waypoint legs |

**So what:** Cost-only routing is not enough. Minimum viable model should consider friction, elevation, vertical factor, and barriers.

---

# Required Solver Behavior

| Behavior | Meaning |
| --- | --- |
| Anisotropic routing | Cost changes with movement direction |
| Arbitrary-direction routing | Path is not locked to grid-neighbor headings |
| Surface-aware distance | Terrain distance and slope are part of cost |
| Server-friendly API | Can run as a repeatable backend job without desktop GIS state |

---

# Why Direction-Independent Solvers Fail

Some solvers allow a cost raster but treat each cell as having one movement cost. That fails when trying to support vertical factors.

Example:

- moving straight uphill/downhill may be "too steep", but it may be acceptable to move tangent to the slope (direction-dependent barrier)
- moving downhill may have a different cost than moving uphill (anisotrpic cost)

A single cell cost cannot express those cases.

**So what:** precomputing slope into a friction raster is not an adequate workaround. The solver itself must evaluate direction.

---

# Why Fixed-Neighbor Graph Solvers Fail

Graph solvers move between a fixed set of neighboring cells.

Typical choices are 4-neighbor, 8-neighbor, or 16-neighbor with knight moves.

That creates:

- stair-step route geometry
- heading bias toward allowed neighbor directions
- distorted costs for routes that should cross cells at arbitrary angles
- poor route quality when the best path is smooth, diagonal, or contour-following

**So what:** graph solvers can be useful baselines, but they are not high-quality route geometry solvers.

---

# Server / Cloud Fit

| Option Type | Operational Fit |
| --- | --- |
| ArcGIS | Technically capable, but ties us to ESRI licensing and deployment patterns |
| GRASS | Can run headlessly in Docker, but "heavier" GIS engine than a simple library AND GPL-licensed (using it creates licensing risk if we link, embed, modify, or redistribute it.)
| Whitebox | Easy to run on server, but incomplete movement model. |
| Python libraries (graph / FMM) | Easy to run on server, but incomplete movement model. |

**So what:** No off-the shelf option is perfect.

---

# Available Alternatives

| Alternative | Missing Features | Anisotropic? | Path Shape | Product Concern |
| --- | --- | --- | --- | --- |
| ArcGIS | None | Yes | Smooth / any-angle | Esri lock-in |
| Whitebox | Elevation + vertical factors + directional slope | No | Direction-biased / zig-zag | Wrong model |
| GRASS `r.walk` | Arbitrary-direction + ArcGIS-style factors | Limited | Direction-biased / zig-zag | Heavy runtime; GPL risk |
| scikit-fmm | Directional slope + route extraction + GIS workflow | No | Smooth accumulation, custom path | Wrong model |
| scikit-image / NetworkX | Continuous routing + built-in mobility model | Custom only | Direction-biased / zig-zag | Custom solver still needed |

**So What:** No off-the-shelf option cleanly matches the full ArcGIS-style vertical-factor routing model plus server-friendly deployment.

---

# What The Custom Solver Should Be

Inputs:

- `cost[row, col]`
- `elevation[row, col]`
- `barrier[row, col]`
- vertical factor function `VF(angle)`
- source and target cells

Local movement cost:

`surface_distance * cell_cost * vertical_factor(slope_angle)`

The important part: `slope_angle` is computed for the proposed movement direction, not stored as one static cell value.

---

# Solver Architecture

Use an ordered-upwind / continuous-front solver rather than a fixed-neighbor graph.

Core idea:

- maintain an accepted front of cells with known minimum accumulated cost
- update candidate cells from nearby accepted points and front segments
- evaluate local movement directions against cost, elevation, barriers, and vertical factor
- store enough direction information to trace a route line back through the solved surface

This gives us ArcGIS-style behavior without depending on ArcGIS as the runtime.

---

# Prototype Features

- cost raster
- elevation raster
- barriers
- ArcGIS-style vertical factor types
- single-source accumulation
- Python bindings
- GeoTIFF / vector adapters
- cropped windows around route legs (to improve performance)
- multi-waypoint routes
- benchmark against ArcGIS, GRASS, Whitebox, and graph baselines

---

# Success Criteria

The prototype is useful if it can demonstrate:

- cost is modelled as `distance × friction × vertical factor`, where the vertical factor is computed from the slope angle of the proposed movement direction.
- tangent-to-slope movement can remain valid when direct uphill movement is blocked
- route geometry is not limited to graph-neighbor headings
- barriers are never crossed
- the solver runs as a normal backend library or service job

---

# Recommended Position

Use off-the-shelf alternatives for comparison/validation.

Leverage FOSS Python GIS stack (`rasterio`, `fiona`, etc.) for handling GIS products, Rust-based core solver for performance, and `maturin` for creating Python bindings.

---

# Reference Notes

- ArcGIS Distance Accumulation supports cost distance, true surface distance, vertical factors, horizontal factors, barriers, and Image Server / Enterprise raster analysis workflows.
- Whitebox CostDistance exposes source + cost/friction inputs and returns accumulation + backlink outputs.
- GRASS `r.walk` supports elevation + friction anisotropic walking-time cost, but uses Dijkstra over raster neighborhoods.
- GRASS can run through command line, Python, and Docker; it is not desktop-only.
- scikit-fmm solves scalar-speed Eikonal travel time on regular grids.

"""
Builds data/cuba.csv for post 2: random points around Cuba, each labelled
land (1) or sea (0), half and half. Also writes data/cuba_outline.json, the
coastline used for the plots.

    python make_cuba.py path/to/ne_50m_admin_0_countries.geojson

Coastline: Natural Earth 1:50m (public domain). Main island and Isla de la
Juventud only.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LON, LAT = (-85.5, -73.5), (19.3, 23.7)
PER_CLASS = 1000

world = json.load(open(sys.argv[1]))
cuba = next(f for f in world["features"] if f["properties"]["ADMIN"] == "Cuba")
rings = sorted((np.array(p[0]) for p in cuba["geometry"]["coordinates"]), key=len)[-2:]   # the two biggest


def inside(pts, ring):
    """Ray casting: count how many coastline edges a ray to the right crosses."""
    x, y = pts[:, :1], pts[:, 1:]
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1)
    return crosses.sum(axis=1) % 2 == 1


rng = np.random.default_rng(0)
pts = np.column_stack([rng.uniform(*LON, 200_000), rng.uniform(*LAT, 200_000)])
land = np.zeros(len(pts), bool)
for ring in rings:
    land |= inside(pts, ring)

keep = np.concatenate([np.flatnonzero(land)[:PER_CLASS], np.flatnonzero(~land)[:PER_CLASS]])
rng.shuffle(keep)
with open(HERE / "cuba.csv", "w") as f:
    f.write("lon,lat,land\n")
    for i in keep:
        f.write(f"{pts[i, 0]:.4f},{pts[i, 1]:.4f},{int(land[i])}\n")
json.dump([r.round(4).tolist() for r in rings], open(HERE / "cuba_outline.json", "w"))
print(f"land fraction of the box: {land.mean():.3f}; wrote {len(keep)} points")

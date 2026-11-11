#!/usr/bin/env python3
"""Merge Q2000 detections across overlapping chips (global NMS in UTM),
add lon/lat, write merged JSON + GeoJSON, and print a comparison summary.

Per-chip NMS cannot merge the same feature seen in two overlapping chips
(e.g. the 0.61 and 0.13 boxes at the same spot in the 600 m site run).
Global NMS on UTM boxes fixes that. IoU threshold matches the run's NMS
(0.1 by default; also report 0.5).

Usage (geo venv):
  merge_detections.py --inputs site12_0.5m:outputs/detections_site12_0.5m_nms01.json \
                      --out outputs/detections_merged.json
"""
import argparse
import json
import os

import numpy as np
from rasterio.warp import transform

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
SITE_E, SITE_N = 699410.8, 2028326.8


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def global_nms(dets, thresh):
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    keep = []
    for i in order:
        a = (dets[i]["e"] - dets[i]["w_m"] / 2, dets[i]["n"] - dets[i]["h_m"] / 2,
             dets[i]["e"] + dets[i]["w_m"] / 2, dets[i]["n"] + dets[i]["h_m"] / 2)
        if all(iou(a, (dets[j]["e"] - dets[j]["w_m"] / 2,
                       dets[j]["n"] - dets[j]["h_m"] / 2,
                       dets[j]["e"] + dets[j]["w_m"] / 2,
                       dets[j]["n"] + dets[j]["h_m"] / 2)) <= thresh
               for j in keep):
            keep.append(i)
    return [dets[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="label:path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nms", type=float, default=0.1)
    args = ap.parse_args()

    merged = {}
    for spec in args.inputs:
        label, path = spec.split(":", 1)
        d = json.load(open(os.path.join(ROOT, path)))
        dets = d["detections"]
        kept = global_nms(dets, args.nms)
        # lon/lat
        es = [r["e"] for r in kept]; ns = [r["n"] for r in kept]
        lons, lats = transform("EPSG:32615", "EPSG:4326", es, ns) if es else ([], [])
        for r, lon, lat in zip(kept, lons, lats):
            r["lon"] = round(lon, 5); r["lat"] = round(lat, 5)
        merged[label] = kept
        near = [r for r in kept if r["dist_site_m"] <= 150]
        print(f"{label}: raw {len(dets)} -> merged {len(kept)} "
              f"(global NMS {args.nms}); within 150 m of site #1: {len(near)}; "
              f"max conf {max([r['conf'] for r in kept], default=0):.2f}")
        for r in sorted(kept, key=lambda r: -r["conf"])[:8]:
            print(f"   conf {r['conf']:.2f} E {r['e']:.0f} N {r['n']:.0f} "
                  f"{r['w_m']:.0f}x{r['h_m']:.0f} m dist {r['dist_site_m']:.0f} m")
    with open(os.path.join(ROOT, args.out), "w") as fh:
        json.dump(merged, fh, indent=2)
    feats = []
    for label, dets in merged.items():
        for r in dets:
            feats.append({"type": "Feature",
                          "properties": {"label": label, "conf": r["conf"],
                                         "w_m": r["w_m"], "h_m": r["h_m"],
                                         "dist_site_m": r["dist_site_m"],
                                         "chip": r["chip"]},
                          "geometry": {"type": "Point",
                                       "coordinates": [r["lon"], r["lat"]]}})
    gj = os.path.join(ROOT, args.out.replace(".json", ".geojson"))
    with open(gj, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=2)
    print("wrote", args.out, "and", gj)


if __name__ == "__main__":
    main()

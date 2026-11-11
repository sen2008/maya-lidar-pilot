#!/usr/bin/env python3
"""Figure: Q2000 detections overlaid on the pilot hillshade.

Panel A: 1200 m site window (composite) with the 0.5 m detections.
Panel B: full-tile downsampled hillshade with all scan detections.
Boxes coloured by confidence; red circle = published site #1 coords.
Run after merge_detections.py and the scan inference (geo venv).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
SITE_E, SITE_N = 699410.8, 2028326.8


def draw_boxes(ax, dets, **kw):
    for d in dets:
        x0 = d["e"] - d["w_m"] / 2
        y0 = d["n"] - d["h_m"] / 2
        ax.add_patch(plt.Rectangle((x0, y0), d["w_m"], d["h_m"],
                                   fill=False, **kw))
        ax.annotate(f"{d['conf']:.2f}", (x0, y0 + d["h_m"]),
                    fontsize=7, color=kw.get("edgecolor", "yellow"),
                    fontweight="bold")


def main():
    merged = json.load(open(os.path.join(ROOT, "outputs/detections_merged.json")))
    scan = json.load(open(os.path.join(ROOT, "outputs/detections_scan_merged.json")))
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Panel A: site 1200 m composite background
    ax = axes[0]
    with rasterio.open(os.path.join(ROOT, "outputs/composite/site12_0.5m/composite.tif")) as src:
        img = src.read([1, 2, 3])
        ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    ax.imshow(np.transpose(img, (1, 2, 0)), extent=ext, origin="upper")
    dets = merged["site12_0.5m"]
    draw_boxes(ax, dets, edgecolor="yellow", lw=1.2)
    ax.plot(SITE_E, SITE_N, marker="o", ms=11, mfc="none", mec="red", mew=2)
    ax.annotate("site #1 (published)", (SITE_E, SITE_N), color="red",
                fontsize=10, fontweight="bold", xytext=(8, 8),
                textcoords="offset points")
    ax.set_title(f"Site window 1200 m, 0.5 m input — {len(dets)} Q2000 detections",
                 fontsize=12)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")

    # Panel B: full tile overview + scan detections
    ax = axes[1]
    with rasterio.open(os.path.join(ROOT, "data/ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif")) as src:
        ov = src.read(1, out_shape=(1024, 1850),
                      resampling=rasterio.enums.Resampling.average)
        ov = np.where(ov > 1e30, np.nan, ov)
        ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    # single-azimuth hillshade on the downsampled DEM (relief reads better
    # than the 4-azimuth average at this scale), contrast-stretched.
    cell = 0.5 * 13872 / 1850
    filled = np.nan_to_num(ov, nan=np.nanmean(ov))
    gy, gx = np.gradient(filled, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    zen = np.radians(45)
    azr = np.radians(360 - 315 + 90)
    hs = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(azr - aspect)
    lo, hi = np.nanpercentile(hs[~np.isnan(ov)], [2, 98])
    hs = np.clip((hs - lo) / (hi - lo), 0, 1)
    ax.imshow(np.where(np.isnan(ov), 1.0, hs), cmap="gray", vmin=0, vmax=1,
              extent=ext, origin="upper")
    dets = scan["scan_0.5m"]
    for d in dets:
        color = "red" if d["conf"] >= 0.5 else "yellow"
        ax.add_patch(plt.Rectangle((d["e"] - d["w_m"] / 2, d["n"] - d["h_m"] / 2),
                                   d["w_m"], d["h_m"], fill=False,
                                   edgecolor=color, lw=0.9))
    ax.plot(SITE_E, SITE_N, marker="o", ms=11, mfc="none", mec="red", mew=2)
    ax.annotate("site #1", (SITE_E, SITE_N), color="red", fontsize=10,
                fontweight="bold", xytext=(8, 8), textcoords="offset points")
    ax.set_title(f"Full tile scan, 0.5 m input — {len(dets)} Q2000 detections "
                 f"(red = conf >= 0.5, {sum(1 for d in dets if d['conf'] >= 0.5)} boxes)",
                 fontsize=12)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs/q2000_detections_figure.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()

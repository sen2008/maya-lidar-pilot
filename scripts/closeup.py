#!/usr/bin/env python3
"""Close-ups: 600x600 m around validation site #1 vs a control patch.

Site #1: UTM15N E 699410.8, N 2028326.8 (Sprajc 2026 complex #1).
Control: a patch ~800 m NW along the same swath with no published site,
centered E 698950, N 2028550 — chosen because it also contains strong
local-relief blobs, so the comparison is fair: does the site look
different from ordinary bumps on the same transect?
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
DEM = os.path.join(ROOT, "data/ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif")
OUT = os.path.join(ROOT, "outputs")
CELL = 0.5
SITE_E, SITE_N = 699410.8, 2028326.8


def hillshade(dem, cell, azimuths=(315, 45, 135, 225), altitude=45):
    gy, gx = np.gradient(dem, cell, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    zen = np.radians(90 - altitude)
    hs = np.zeros_like(dem, dtype=np.float32)
    for az in azimuths:
        azr = np.radians(360 - az + 90)
        hs += (np.cos(zen) * np.cos(slope)
               + np.sin(zen) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(hs / len(azimuths), 0, 1)


def get_window(src, ec, nc, size_m=600):
    e0, e1 = ec - size_m / 2, ec + size_m / 2
    n0, n1 = nc - size_m / 2, nc + size_m / 2
    win = rasterio.windows.from_bounds(e0, n0, e1, n1, src.transform)
    dem = src.read(1, window=win, boundless=True,
                   fill_value=1.70141e38).astype(np.float32)
    nodata = dem > 1e30
    fill = np.nanmean(np.where(nodata, np.nan, dem))
    dem_f = np.where(nodata, fill, dem)
    hs = hillshade(dem_f, CELL)
    relief = dem_f - ndimage.gaussian_filter(dem_f, sigma=50)
    relief = np.where(nodata, 0, relief)
    return dem_f, hs, relief, nodata, (e0, e1, n0, n1)


def main():
    with rasterio.open(DEM) as src:
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        for (ax_hs, ax_rl, (ec, nc, tag)) in zip(
                [axes[0, 0], axes[1, 0]], [axes[0, 1], axes[1, 1]],
                [(SITE_E, SITE_N, "SITE #1"),
                 (698950, 2028550, "CONTROL")]):
            dem, hs, relief, nodata, ext = get_window(src, ec, nc)
            print(f"{tag}: nodata frac {nodata.mean():.3f}, "
                  f"relief range {relief.min():.2f}..{relief.max():.2f} m, "
                  f"frac >0.4m: {(relief > 0.4).mean():.4f}")
            ax_hs.imshow(np.where(nodata, np.nan, hs), cmap="gray",
                         vmin=0.2, vmax=0.95, extent=ext, origin="upper")
            ax_hs.set_title(f"{tag} — hillshade (600 m)", fontsize=12)
            ax_rl.imshow(relief, cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                         extent=ext, origin="upper")
            ax_rl.set_title(f"{tag} — local relief, sigma 25 m (m)", fontsize=12)
            if tag == "SITE #1":
                for ax in (ax_hs, ax_rl):
                    ax.plot(SITE_E, SITE_N, marker="o", ms=10, mfc="none",
                            mec="red", mew=2)
                    ax.annotate("published coords", (SITE_E, SITE_N),
                                color="red", fontsize=10, fontweight="bold",
                                xytext=(8, 8), textcoords="offset points")
        for ax in axes.flat:
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "closeup_site_vs_control.png"), dpi=120)
        print("wrote", os.path.join(OUT, "closeup_site_vs_control.png"))


if __name__ == "__main__":
    main()

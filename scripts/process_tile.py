#!/usr/bin/env python3
"""Process one G-LiHT DEM tile: hillshade / slope / local relief + crude screening.

Tile: ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif (0.5 m DTM, EPSG:32615).
Validation site: Sprajc 2026 nested complex #1, 18.3354 N, 91.1130 W
  -> UTM15N E 699410.8, N 2028326.8.

Method (explicitly crude — no trained model exists here):
  * hillshade: mean of 4 azimuths (315/45/135/225 deg), 45 deg altitude.
  * slope: degrees from the DEM gradient.
  * local relief: DTM minus Gaussian-smoothed DTM (sigma 25 m).
    Positive local relief = ground standing proud of its surroundings:
    mounds, platforms, causeways — and natural rises.
  * screening: threshold local relief > 0.4 m, label connected components,
    keep 50–5000 m^2, report centroid lon/lat, area, mean relief, bbox
    aspect. Validation = a flagged component at the known site.

Memory: the full tile is 13872x7686 float32 (~426 MB on disk). We never
load it whole; we read a 2x2 km window around the site and a downsampled
overview. Run with the pilot venv: venv/bin/python scripts/process_tile.py
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
DEM = os.path.join(ROOT, "data/ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

SITE_LON, SITE_LAT = -91.1130, 18.3354
SITE_E, SITE_N = 699410.8, 2028326.8  # UTM15N, verified via rasterio.warp.transform
CELL = 0.5  # m/px


def hillshade(dem, cell, azimuths=(315, 45, 135, 225), altitude=45):
    """Mean multi-direction hillshade, 0..255. dem in metres, nodata masked."""
    gy, gx = np.gradient(dem, cell, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # aspect measured from north, clockwise-ish
    zen = np.radians(90 - altitude)
    hs = np.zeros_like(dem, dtype=np.float32)
    for az in azimuths:
        azr = np.radians(360 - az + 90)  # convert compass azimuth to math angle
        hs += (np.cos(zen) * np.cos(slope)
               + np.sin(zen) * np.sin(slope) * np.cos(azr - aspect))
    hs /= len(azimuths)
    return np.clip(hs, 0, 1)


def plot(arr, path, title, cmap="gray", vmin=None, vmax=None, markers=(),
         extent=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
              origin="upper")
    for (e, n, label, color) in markers:
        ax.plot(e, n, marker="o", ms=9, mfc="none", mec=color, mew=2)
        ax.annotate(label, (e, n), color=color, fontsize=11,
                    fontweight="bold", xytext=(6, 6), textcoords="offset points")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Easting (m, UTM 15N)")
    ax.set_ylabel("Northing (m, UTM 15N)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    with rasterio.open(DEM) as src:
        print(f"tile {src.width}x{src.height} {src.crs} bounds={src.bounds}")
        # --- overview: downsampled hillshade of the whole tile ---
        ov = src.read(1, out_shape=(1024, 1850), resampling=rasterio.enums.Resampling.average)
        ov = np.where(ov > 1e30, np.nan, ov)
        hs_ov = hillshade(np.nan_to_num(ov, nan=np.nanmean(ov)), CELL * src.width / 1850)
        plot(hs_ov, os.path.join(OUT, "overview_hillshade.png"),
             "l2s502 — full tile hillshade (downsampled). Red = validation site #1",
             markers=[(SITE_E, SITE_N, "site #1", "red")],
             extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
        # --- 2x2 km window around the site ---
        e0, e1 = SITE_E - 1000, SITE_E + 1000
        n0 = max(src.bounds.bottom, SITE_N - 1000)
        n1 = min(src.bounds.top, SITE_N + 1000)
        win = rasterio.windows.from_bounds(e0, n0, e1, n1, src.transform)
        dem = src.read(1, window=win).astype(np.float32)
        nodata = dem > 1e30
        print(f"window {dem.shape}, nodata frac {nodata.mean():.4f}, "
              f"elev range {np.nanmin(np.where(nodata, np.nan, dem)):.1f}–"
              f"{np.nanmax(np.where(nodata, np.nan, dem)):.1f} m")
        fill = np.nanmean(np.where(nodata, np.nan, dem))
        dem_f = np.where(nodata, fill, dem)
        hs = hillshade(dem_f, CELL)
        slope = np.degrees(np.arctan(np.hypot(*np.gradient(dem_f, CELL, CELL))))
        smooth = ndimage.gaussian_filter(dem_f, sigma=50)  # 50 px = 25 m
        relief = dem_f - smooth
        relief = np.where(nodata, 0, relief)

        xs = np.linspace(e0, e1, dem.shape[1])
        ys = np.linspace(n1, n0, dem.shape[0])  # rows go top->bottom
        ext = (e0, e1, n0, n1)
        plot(hs, os.path.join(OUT, "window_hillshade.png"),
             "l2s502 — 2x2 km window hillshade. Red = validation site #1",
             markers=[(SITE_E, SITE_N, "site #1", "red")], extent=ext)
        plot(slope, os.path.join(OUT, "window_slope.png"),
             "l2s502 — 2x2 km window slope (deg)",
             cmap="viridis", vmin=0, vmax=30,
             markers=[(SITE_E, SITE_N, "site #1", "red")], extent=ext)
        plot(relief, os.path.join(OUT, "window_relief.png"),
             "l2s502 — 2x2 km local relief, sigma 25 m (m). Red = validation site #1",
             cmap="RdBu_r", vmin=-1.5, vmax=1.5,
             markers=[(SITE_E, SITE_N, "site #1", "red")], extent=ext)

        # --- crude screening on local relief ---
        mask = relief > 0.4
        lab, ncomp = ndimage.label(mask)
        print(f"components above 0.4 m relief: {ncomp}")
        rows = []
        for i in range(1, ncomp + 1):
            ys_i, xs_i = np.where(lab == i)
            area = len(xs_i) * CELL * CELL
            if not (50 <= area <= 5000):
                continue
            e_c = e0 + xs_i.mean() * CELL
            n_c = n1 - ys_i.mean() * CELL
            h, w = ys_i.max() - ys_i.min() + 1, xs_i.max() - xs_i.min() + 1
            aspect = max(h, w) / max(1, min(h, w))
            lon, lat = transform("EPSG:32615", "EPSG:4326", [e_c], [n_c])
            rows.append(dict(id=i, lon=lon[0], lat=lat[0], e=e_c, n=n_c,
                             area=area, mean_relief=float(relief[lab == i].mean()),
                             aspect=float(aspect),
                             dist_site=float(np.hypot(e_c - SITE_E, n_c - SITE_N))))
        rows.sort(key=lambda r: r["dist_site"])
        # mark all screened components on the relief plot
        marks = [(r["e"], r["n"], f"#{r['id']}", "yellow") for r in rows[:12]]
        marks.append((SITE_E, SITE_N, "site #1", "red"))
        plot(relief, os.path.join(OUT, "window_relief_marked.png"),
             "l2s502 — local relief with screened components (yellow) and site #1 (red)",
             cmap="RdBu_r", vmin=-1.5, vmax=1.5, markers=marks, extent=ext)
        with open(os.path.join(OUT, "candidates.md"), "w") as fh:
            fh.write("# Screening results — l2s502, 2x2 km window around site #1\n\n")
            fh.write("Crude screening: local relief (DTM minus 25 m Gaussian) > 0.4 m, "
                     "connected components 50–5000 m^2. No trained model; "
                     "false positives expected.\n\n")
            fh.write(f"Validation site #1: 18.3354 N, 91.1130 W "
                     f"(UTM {SITE_E:.1f} E, {SITE_N:.1f} N).\n\n")
            fh.write("| id | lat | lon | area m2 | mean relief m | aspect | dist to site #1 (m) |\n")
            fh.write("|---|---|---|---|---|---|---|\n")
            for r in rows:
                fh.write(f"| {r['id']} | {r['lat']:.5f} | {r['lon']:.5f} | "
                         f"{r['area']:.0f} | {r['mean_relief']:.2f} | "
                         f"{r['aspect']:.2f} | {r['dist_site']:.0f} |\n")
            fh.write(f"\nTotal components kept: {len(rows)} (of {ncomp} above threshold).\n")
            near = [r for r in rows if r["dist_site"] < 150]
            fh.write(f"\nComponents within 150 m of site #1: {len(near)} — "
                     "validation passes if >= 1 lands on the published complex.\n")
        print(f"kept {len(rows)} components; within 150 m of site: "
              f"{len([r for r in rows if r['dist_site'] < 150])}")
        for r in rows[:10]:
            print(f"  #{r['id']} lat {r['lat']:.5f} lon {r['lon']:.5f} "
                  f"area {r['area']:.0f} m2 relief {r['mean_relief']:.2f} m "
                  f"dist {r['dist_site']:.0f} m")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the Q2000 input composite from the pilot DEM (Table 3 of Britton et al. 2026).

The model was NOT trained on a plain hillshade. Input is a 4-layer Relief
Visualization Toolkit composite, layered in ArcGIS Pro per Table 3:

  Layer  Visualization   Transparency  Blend       Brightness  Contrast  Gamma
  1       Multi-hillshade  (base)        Normal      10          20        2
  2       Slope            33            Difference  10          20        3
  3       Open-Positive    33            Overlay     -30         55        1
  4       Skyview          33            Multiply    1           30        0.5

  "Process settings (all layers stretched to Min/Max)."

Reproduction choices (recorded ambiguities — the paper does not specify them):
  * RVT generation parameters are NOT in the paper (author used the RVT GUI
    "to satisfaction"). We use RVT_py defaults: multi-hillshade 16 dirs,
    sun elevation 35 deg; slope in degrees; SVF/openness 16 dirs,
    search radius 10 px, noise removal 0, ve_factor 1.
  * ArcGIS brightness/contrast/gamma math is undocumented; we use the
    standard reading: stretch to [0,1], gamma as v**(1/gamma) (gamma>1
    brightens, matching ArcGIS docs), contrast as (v-0.5)*(1+c/100)+0.5,
    brightness as v+b/100, clip to [0,1]. Order: gamma, contrast, brightness.
  * Blend with layer transparency (alpha = 1 - transparency/100):
    out = (1-a)*base + a*blend(base, top). Standard GIMP/Photoshop semantics.
  * Min/max stretch is per processed window (paper used whole-transect
    rasters; our windows are 600 m / 2 km patches — recorded, not hidden).
  * Nodata (outside the G-LiHT swath) is filled with the window mean before
    RVT, exactly as the pilot did; chips >50% nodata are skipped at export.

Usage (geo venv: venv/bin/python):
  make_composite.py --center 699410.8,2028326.8 --size-m 600 --res 0.5 --tag site
  make_composite.py --center 698950,2028550 --size-m 600 --res 0.5 --tag control
  make_composite.py --center 699410.8,2028326.8 --size-m 600 --res 1.0 --tag site1m

Outputs (outputs/composite/<tag>_<res>m/):
  composite.tif   3-band uint8 RGB GeoTIFF
  composite.png   preview
  chips/chip_*.npy  416x416 uint8 chips, stride 208 (ArcGIS padding 104)
  chips.json      manifest: chip id, UTM origin, nodata fraction
  run.json        parameters + ambiguity record
"""
import argparse
import json
import os
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import rvt.vis as rvt

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
DEM = os.path.join(ROOT, "data/ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif")
OUTROOT = os.path.join(ROOT, "outputs", "composite")

# Table 3: (visualization, transparency %, blend, brightness, contrast, gamma)
LAYERS = [
    ("mhs", 0, "normal", 10, 20, 2.0),
    ("slope", 33, "difference", 10, 20, 3.0),
    ("opns", 33, "overlay", -30, 55, 1.0),
    ("svf", 33, "multiply", 1, 30, 0.5),
]


def adjust(v, brightness, contrast, gamma):
    """ArcGIS-style brightness/contrast/gamma on [0,1] values (see header)."""
    v = np.clip(v, 0.0, 1.0)
    if gamma != 1.0:
        v = np.power(v, 1.0 / gamma)
    if contrast != 0:
        v = (v - 0.5) * (1.0 + contrast / 100.0) + 0.5
    if brightness != 0:
        v = v + brightness / 100.0
    return np.clip(v, 0.0, 1.0)


def blend(base, top, mode):
    if mode == "normal":
        return top
    if mode == "multiply":
        return base * top
    if mode == "overlay":
        return np.where(base < 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    if mode == "difference":
        return np.abs(base - top)
    raise ValueError(mode)


def stretch01(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        return np.zeros_like(v), lo, hi
    return (v - lo) / (hi - lo), lo, hi


def downsample_mean(dem, nodata, factor=2):
    """Block-mean downsample by `factor`, ignoring nodata pixels."""
    h, w = dem.shape
    h2, w2 = h // factor, w // factor
    dem = dem[: h2 * factor, : w2 * factor]
    nodata = nodata[: h2 * factor, : w2 * factor]
    blocks = dem.reshape(h2, factor, w2, factor)
    mask = nodata.reshape(h2, factor, w2, factor)
    with np.errstate(invalid="ignore"):
        out = np.where(mask, np.nan, blocks)
        m = np.nanmean(out, axis=(1, 3))
    new_nodata = np.isnan(m)
    return m, new_nodata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, help="E,N (UTM 15N)")
    ap.add_argument("--size-m", type=float, default=600)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    ec, nc = [float(x) for x in args.center.split(",")]
    cell = args.res
    size = args.size_m
    outdir = os.path.join(OUTROOT, f"{args.tag}_{args.res:g}m")
    os.makedirs(os.path.join(outdir, "chips"), exist_ok=True)

    e0, e1 = ec - size / 2, ec + size / 2
    n0, n1 = nc - size / 2, nc + size / 2
    t0 = time.time()
    with rasterio.open(DEM) as src:
        win = rasterio.windows.from_bounds(e0, n0, e1, n1, src.transform)
        dem = src.read(1, window=win, boundless=True,
                       fill_value=1.70141e38).astype(np.float64)
        transform = src.window_transform(win)
    nodata = dem > 1e30
    print(f"window {dem.shape} nodata frac {nodata.mean():.3f}")
    if cell != 0.5:
        dem, nodata = downsample_mean(dem, nodata, factor=int(round(cell / 0.5)))
        print(f"downsampled to {cell} m: {dem.shape} nodata frac {nodata.mean():.3f}")
    fill = float(np.nanmean(np.where(nodata, np.nan, dem)))
    dem_f = np.where(nodata, fill, dem).astype(np.float32)  # float32 halves RVT memory

    # --- RVT layers (defaults; see header) ---
    # NOTE: multi_hillshade returns the STACK of 16 directional hillshades
    # (16,h,w); the RVT "MultiHillshade" visualization is their mean. We
    # average single-direction hillshades one at a time instead of stacking
    # (a 4000x4000 stack would be ~1 GB). hillshade() also drops the outer
    # 1-px border on all sides, so the other layers are cropped [1:-1, 1:-1]
    # to align, and the window origin shifts by one pixel.
    t = time.time()
    sa = rvt.slope_aspect(dem_f, cell, cell, output_units="radian",
                          ve_factor=1, no_data=np.nan)
    slope_rad, aspect_rad = sa["slope"], sa["aspect"]
    mhs = np.zeros((dem_f.shape[0] - 2, dem_f.shape[1] - 2), dtype=np.float64)
    for i in range(16):
        az = (360 / 16) * i
        mhs += rvt.hillshade(dem_f, cell, cell, sun_azimuth=az, sun_elevation=35,
                             slope=slope_rad, aspect=aspect_rad,
                             ve_factor=1, no_data=np.nan)
    mhs /= 16.0
    # hillshade drops the border: crop mhs back is unnecessary — it is
    # already (h-2, w-2); crop the others to match.
    print(f"multi_hillshade {time.time()-t:.1f}s mean of 16 dirs, "
          f"range {np.nanmin(mhs):.3f}..{np.nanmax(mhs):.3f}")
    t = time.time()
    slope = np.degrees(slope_rad)
    print(f"slope {time.time()-t:.1f}s range "
          f"{np.nanmin(slope):.2f}..{np.nanmax(slope):.2f} deg")
    t = time.time()
    svf_out = rvt.sky_view_factor(dem_f, cell, compute_svf=True, compute_opns=True,
                                  svf_n_dir=16, svf_r_max=10, svf_noise=0,
                                  ve_factor=1, no_data=np.nan)
    svf, opns = svf_out["svf"], svf_out["opns"]
    print(f"svf/opns {time.time()-t:.1f}s svf {np.nanmin(svf):.3f}..{np.nanmax(svf):.3f} "
          f"opns {np.nanmin(opns):.2f}..{np.nanmax(opns):.2f} deg")
    crop = slice(1, -1)
    slope, svf, opns = slope[crop, crop], svf[crop, crop], opns[crop, crop]
    nodata = nodata[crop, crop]
    e0, n1 = e0 + cell, n1 - cell
    e1, n0 = e0 + mhs.shape[1] * cell, n1 - mhs.shape[0] * cell

    layers = {"mhs": mhs, "slope": slope, "opns": opns, "svf": svf}
    comp = None
    stats = {}
    for name, transp, mode, bright, contr, gamma in LAYERS:
        v, lo, hi = stretch01(layers[name])
        v = adjust(v, bright, contr, gamma)
        stats[name] = {"min": lo, "max": hi, "blend": mode,
                       "transparency": transp, "brightness": bright,
                       "contrast": contr, "gamma": gamma}
        if comp is None:
            comp = v
        else:
            a = 1.0 - transp / 100.0
            comp = (1 - a) * comp + a * blend(comp, v, mode)
    # Table 3 produces a single-layer visualization; ArcGIS exports it as a
    # 3-channel RGB GeoTIFF (same grayscale in R, G, B) — replicate that.
    gray = np.clip(comp * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    print(f"composite range {gray.min()}..{gray.max()} mean {gray.mean():.1f}")

    # --- write GeoTIFF (origin = cropped window top-left) ---
    h, w = rgb.shape[:2]
    tr = rasterio.Affine(cell, 0.0, e0, 0.0, -cell, n1)
    prof = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8",
                crs="EPSG:32615", transform=tr, compress="lzw")
    with rasterio.open(os.path.join(outdir, "composite.tif"), "w", **prof) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)
    print("wrote", os.path.join(outdir, "composite.tif"))

    # --- preview ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, extent=(e0, e1, n0, n1), origin="upper")
    ax.plot(ec, nc, marker="o", ms=10, mfc="none", mec="red", mew=2)
    ax.set_title(f"Q2000 composite {args.tag} {args.res:g} m "
                 f"(RVT Table 3, {size:.0f} m window)")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "composite.png"), dpi=110)
    plt.close(fig)

    # --- chips: 416x416, stride 208 (ArcGIS padding 104) ---
    chips = []
    stride = 208
    n_chips = 0
    for r0 in range(0, h - 416 + 1, stride):
        for c0 in range(0, w - 416 + 1, stride):
            nd_frac = float(nodata[r0:r0 + 416, c0:c0 + 416].mean())
            if nd_frac > 0.5:
                continue
            chip = rgb[r0:r0 + 416, c0:c0 + 416]
            cid = f"{args.tag}_{args.res:g}m_r{r0}_c{c0}"
            np.save(os.path.join(outdir, "chips", cid + ".npy"), chip)
            chips.append({"id": cid, "tag": args.tag, "res": args.res,
                          "row": r0, "col": c0,
                          "e_left": e0 + c0 * cell,
                          "n_top": n1 - r0 * cell,
                          "cell": cell, "nodata_frac": round(nd_frac, 3)})
            n_chips += 1
    with open(os.path.join(outdir, "chips.json"), "w") as fh:
        json.dump({"window": {"e0": e0, "e1": e1, "n0": n0, "n1": n1,
                              "center_e": ec, "center_n": nc},
                   "cell": cell, "chip_size": 416, "stride": stride,
                   "chips": chips}, fh, indent=2)
    run = {"tag": args.tag, "res": cell, "window_m": size,
           "rvt_defaults": {"mhs_dirs": 16, "mhs_sun_el": 35,
                            "svf_dirs": 16, "svf_r_max_px": 10,
                            "svf_noise": 0, "slope_units": "degree"},
           "layer_stats": stats,
           "ambiguities": [
               "RVT generation parameters not in paper; RVT_py defaults used.",
               "ArcGIS brightness/contrast/gamma math undocumented; "
               "gamma as v**(1/gamma), contrast (v-0.5)*(1+c/100)+0.5, "
               "brightness v+b/100; order gamma, contrast, brightness.",
               "Min/max stretch per window, not per whole transect.",
               "Nodata filled with window mean before RVT.",
           ],
           "n_chips": n_chips, "nodata_frac": float(nodata.mean()),
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(outdir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=2)
    print(f"chips: {n_chips} in {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()

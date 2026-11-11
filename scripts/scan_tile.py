#!/usr/bin/env python3
"""Full-tile Q2000 scan: block-wise RVT composite + chip export for one tile.

Same Table 3 recipe as make_composite.py (imported), but tiled over the
whole DEM so the scan covers the full valid swath, not just a window.

Blocking: 2080x2080 px blocks (2080 = 10x the 208 chip stride, so chip
grids align across blocks; chips fully inside one block, no gaps or
duplicates). Each block is read with a 64 px halo for RVT context, then
cropped back to the central block. Two passes:
  pass 1: compute the 4 RVT layers per block, save float32 .npy, track
          global per-layer min/max (all pixels, filled nodata included).
  pass 2: stretch with the GLOBAL min/max, adjust, blend, export chips.
Nodata is filled with the block mean before RVT (as in the pilot).

Usage (geo venv):
  scan_tile.py --res 0.5 --outdir outputs/scan_0.5m
Outputs: <outdir>/layers/block_*.npy, chips/*.npy, chips.json, run.json.
Chips are then run through infer.py like any other manifest.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import rvt.vis as rvt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_composite import adjust, blend, stretch01, LAYERS

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
DEM = os.path.join(ROOT, "data/ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif")
BLOCK = 2080
HALO = 64
CHIP = 416
STRIDE = 208


def rvt_layers(dem_f, cell):
    """Return dict of the 4 RVT layers (float32/64 arrays, same shape as dem_f
    except mhs which is (h-2, w-2))."""
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
    svf_out = rvt.sky_view_factor(dem_f, cell, compute_svf=True, compute_opns=True,
                                  svf_n_dir=16, svf_r_max=10, svf_noise=0,
                                  ve_factor=1, no_data=np.nan)
    return {"mhs": mhs, "slope": np.degrees(slope_rad),
            "svf": svf_out["svf"], "opns": svf_out["opns"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cell = args.res
    scale = int(round(cell / 0.5))  # tile px per scan px
    outdir = os.path.join(ROOT, args.outdir)
    os.makedirs(os.path.join(outdir, "layers"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "chips"), exist_ok=True)
    t0 = time.time()

    with rasterio.open(DEM) as src:
        W, H = src.width, src.height
        transform = src.transform
        print(f"tile {W}x{H} cell 0.5 m; scan at {cell} m")
        # block grid (tile coords, 0.5 m px)
        blocks = []
        for br in range(0, H, BLOCK):
            for bc in range(0, W, BLOCK):
                bh = min(BLOCK, H - br)
                bw = min(BLOCK, W - bc)
                blocks.append((br, bc, bh, bw))
        print(f"{len(blocks)} blocks")

        # ---- pass 1: RVT layers per block + global min/max ----
        gmin, gmax = {}, {}
        kept_blocks = []
        for (br, bc, bh, bw) in blocks:
            r0, c0 = max(0, br - HALO), max(0, bc - HALO)
            r1, c1 = min(H, br + bh + HALO), min(W, bc + bw + HALO)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            dem = src.read(1, window=win).astype(np.float64)
            nodata = dem > 1e30
            if nodata.mean() > 0.98:
                continue
            if cell != 0.5:
                # downsample block-mean by `scale` (valid pixels only)
                h2, w2 = dem.shape[0] // scale, dem.shape[1] // scale
                dem = dem[: h2 * scale, : w2 * scale]
                nodata = nodata[: h2 * scale, : w2 * scale]
                with np.errstate(invalid="ignore"):
                    m = np.nanmean(np.where(nodata, np.nan,
                                            dem.reshape(h2, scale, w2, scale)),
                                   axis=(1, 3))
                nodata = np.isnan(m)
                dem = m
            fill = float(np.nanmean(np.where(nodata, np.nan, dem)))
            dem_f = np.where(nodata, fill, dem).astype(np.float32)
            layers = rvt_layers(dem_f, cell)
            # hillshade drops the outer 1-px border; edge-pad mhs back to
            # full size so all layers share coordinates (1 px of edge
            # replication at block/tile edges is harmless).
            layers["mhs"] = np.pad(layers["mhs"], 1, mode="edge")
            # halo rows/cols actually read above/left of the block (tile px)
            h0, w0 = br - r0, bc - c0
            # crop to central block, in scan-resolution px
            h0c, w0c = h0 // scale, w0 // scale
            bh_s, bw_s = bh // scale, bw // scale
            central = {}
            for name, arr in layers.items():
                central[name] = arr[h0c: h0c + bh_s, w0c: w0c + bw_s]
            # nodata mask for the central block (at scan resolution)
            nd = nodata[h0c: h0c + bh_s, w0c: w0c + bw_s]
            tag = f"b{br}_{bc}"
            for name, arr in central.items():
                np.save(os.path.join(outdir, "layers", f"{tag}_{name}.npy"),
                        arr.astype(np.float32))
                lo, hi = float(np.min(arr)), float(np.max(arr))
                gmin[name] = min(gmin.get(name, lo), lo)
                gmax[name] = max(gmax.get(name, hi), hi)
            np.save(os.path.join(outdir, "layers", f"{tag}_nodata.npy"), nd)
            kept_blocks.append((br, bc, bh, bw, tag))
            print(f"  block {tag} {bh}x{bw} nodata {nd.mean():.2f} "
                  f"({time.time()-t0:.0f}s)")
        print(f"pass 1 done: {len(kept_blocks)} blocks with data, "
              f"global min/max: {gmin} / {gmax}")

        # ---- pass 2: composite + chips ----
        chips = []
        for (br, bc, bh, bw, tag) in kept_blocks:
            central = {n: np.load(os.path.join(outdir, "layers", f"{tag}_{n}.npy"))
                       for n in ("mhs", "slope", "opns", "svf")}
            nd = np.load(os.path.join(outdir, "layers", f"{tag}_nodata.npy"))
            comp = None
            for name, transp, mode, bright, contr, gamma in LAYERS:
                v = (central[name].astype(np.float64) - gmin[name]) / \
                    (gmax[name] - gmin[name])
                v = adjust(v, bright, contr, gamma)
                if comp is None:
                    comp = v
                else:
                    a = 1.0 - transp / 100.0
                    comp = (1 - a) * comp + a * blend(comp, v, mode)
            rgb = np.repeat(np.clip(comp * 255, 0, 255).astype(np.uint8)[:, :, None],
                            3, axis=2)
            # chips fully inside this block, on the global 208 grid.
            # (br, bc, bh, bw) are tile px; block size in scan px:
            bh_s, bw_s = bh // scale, bw // scale
            for r0 in range(0, bh_s - CHIP + 1, STRIDE):
                for c0 in range(0, bw_s - CHIP + 1, STRIDE):
                    if nd[r0:r0 + CHIP, c0:c0 + CHIP].mean() > 0.5:
                        continue
                    chip = rgb[r0:r0 + CHIP, c0:c0 + CHIP]
                    cid = f"scan_{cell:g}m_r{br + r0 * scale}_c{bc + c0 * scale}"
                    np.save(os.path.join(outdir, "chips", cid + ".npy"), chip)
                    e_left = transform.c + (bc + c0 * scale) * 0.5
                    n_top = transform.f - (br + r0 * scale) * 0.5
                    chips.append({"id": cid, "tag": f"scan_{cell:g}m", "res": cell,
                                  "row": br + r0 * scale, "col": bc + c0 * scale,
                                  "e_left": e_left, "n_top": n_top,
                                  "cell": cell,
                                  "nodata_frac": round(float(
                                      nd[r0:r0 + CHIP, c0:c0 + CHIP].mean()), 3)})
            print(f"  block {tag}: chips so far {len(chips)}")
    with open(os.path.join(outdir, "chips.json"), "w") as fh:
        json.dump({"cell": cell, "chip_size": CHIP, "stride": STRIDE,
                   "blocks": len(kept_blocks), "global_min": gmin,
                   "global_max": gmax, "chips": chips}, fh, indent=2)
    print(f"scan done: {len(chips)} chips in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

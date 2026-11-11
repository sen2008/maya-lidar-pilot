# Pilot log

## 2026-11-11 — Phase 0: data access (DONE, no login needed)

- G-LiHT is distributed via LP DAAC (GLDTMK.001, doi:10.5067/Community/
  GLIHT/GLDTMK.001); the canonical route needs a free NASA Earthdata
  login. **Not needed for this pilot:** an open mirror exists.
- **Open mirror found:** Scholar@UC (Univ. of Cincinnati) 2026
  reprocessing of the original G-LiHT LAS files — per-region/per-flight
  ground-filtered LAZ + 50 cm / 33 cm / 100 cm DEM GeoTIFFs, plus G1
  layered visualizations. Downloads work with no auth (ranged GET
  returns 206; HEAD returns 500 on that host — use GET).
- G-LiHT Mexico coverage is 2013 AMIGACarb transects across the
  Yucatan: regions South, North2, Centro, Norte, Chiapas, Campeche.
- Valeriana is NOT a G-LiHT validation site (found in 2013 TNC
  environmental LiDAR, not G-LiHT). Validation site chosen instead:
  Šprajc (2026) nested complex #1, 18.3354° N, 91.1130° W, published
  as detected on G-LiHT transect I2s502 — which is this tile.
- Environment: venv at `venv/` (system numpy is Debian-owned and pip
  refuses to upgrade it in place): rasterio 1.5.2 / GDAL 3.12.2,
  numpy 2.5.4, laspy 2.7.0, matplotlib 3.11.2.
- VM quirk: GDAL's bundled curl rejects the egress proxy's TLS
  (self-signed cert in chain). Fix for /vsicurl/ reads:
  `GDAL_HTTP_UNSAFESSL=YES CPL_VSIL_CURL_USE_HEAD=NO`. Plain curl
  works (uses the proxy CA bundle from env).

## 2026-11-11 — Tile selected and downloading

- Tile: `ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif`, 426,482,130 bytes,
  13,872 x 7,686 px @ 0.5 m, EPSG:32615. Bounds verified via /vsicurl/
  before downloading. Site #1 confirmed inside the tile.
- Download started 02:02 UTC, ~1.5 MB/s, ETA ~5 min.
- Download complete 02:04 UTC: 426,482,130 bytes, exactly the
  advertised size. Raw LiDAR stays local only (never pushed).

## 2026-11-11 — Processing (DONE)

- Script: `scripts/process_tile.py` (venv python). Reads the DEM in
  windows (never loads the full 13872x7686 tile; ~655 MB RAM free).
- Products in `outputs/`: full-tile hillshade + elevation overview,
  2x2 km window hillshade/slope/local-relief (sigma 25 m), marked
  relief plot, 600 m close-up site-vs-control.
- Window around site #1: 3515x4000 px, nodata fraction 0.787 —
  the G-LiHT transect is a narrow diagonal swath; most of the
  2x2 km window is outside the flight line. Elevation 45.0–58.1 m.
- Screening: local relief > 0.4 m gave 1505 components, 58 kept
  at 50–5000 m², 12 within 150 m of site #1.

## 2026-11-11 — Validation verdict: PARTIAL PASS

- HILLSHADE (works): rectilinear platform outlines visible within
  ~100 m of the published coords — a square platform at ~699300 E,
  2028470 N, a rectangular outline at ~699600 E, 2028330 N, and a
  cluster of small rings NW of the marker. Matches Šprajc's
  description of nested complex #1. A straight diagonal line
  (possible causeway or modern track) also crosses the window —
  unidentified.
- AUTOMATED SCREENING (fails): site window 7.9% pixels > 0.4 m
  relief vs 6.0% in a control window; relief maxima not at the
  site; blobs mostly round in both windows. Height threshold
  cannot separate low platforms from natural karst bumps.
- Lesson: visual hillshade inspection is the working method (as in
  the literature); automation needs shape-based screening or a
  trained model (Q2000 CNN, F1 0.89 in the 2026 Springer paper).
- No new site claimed. The only literature-backed feature in the
  tile is the already-known complex #1.

## 2026-11-11 — Q2000 validation run (Lucas-approved)

- RVT composite reproduced from paper Table 3 (MultiHillshade + Slope +
  Open-Positive + Skyview) with RVT_py. Ambiguities recorded in RESULTS.md:
  RVT params not in paper (16 dirs, sun elev 35 deg, SVF radius 10 px);
  ArcGIS brightness/contrast/gamma math undocumented (assumed gamma ->
  contrast -> brightness, order and formulas recorded); stretch is
  per-window for window runs, per-tile for the full scan.
- YOLOv3 DarkNet53 built to match the state dict; STRICT LOAD PASSES
  (missing [] unexpected []). Two cfg-route bugs fixed during the build
  (module 15 input = module 12, module 23 input = module 20).
- VERDICT: the known site IS detected. Window runs (per-window stretch):
  strongest box 0.61 (600 m) / 0.69 (1200 m) at 62-64 m from published
  coords, 50x48 m; 8 boxes within 150 m (600 m), 6 (1200 m). Full-tile
  scan (global stretch): 42 detections, 5 within 150 m, top 0.37 @ 101 m.
- 0.5 m BEATS 1 m: same 1200 m window, 24 detections (max 0.93) vs 13
  (max 0.72). Use 0.5 m (matches .emd cell size 0.513 m).
- NMS 0.1 vs 0.5 barely matters (24 vs 28 at 0.5 m; site detected at both).
- Control 600 m: 4 detections, max 0.56 — but control is not a true
  negative (dense settlement; its top feature re-detects at 0.81 in the
  scan, likely a real structure). No false-positive rate computable.
- CAUTION: at any practical threshold (>= 0.5) the full-tile scan would
  MISS the known site (0.37 max near site). Stretch choice moves the
  site's confidence 0.37 (global) -> 0.69 (per-window). F1 0.89 does not
  obviously replicate; package val AP ~0.62 is the consistent number.
  Model = candidate generator needing human review, not a trusted detector.
- Top unvetted candidates away from site #1: 0.90 @ 699081 E, 2028377 N;
  0.86 @ 698441 E, 2028526 N; 0.81 @ 699064 E, 2028569 N. Not claimed.
- Full report: RESULTS.md. Figure: outputs/q2000_detections_figure.png.
- Scale-up: fix stretch convention + re-validate; need labelled ground
  truth for any F1 claim; ~200 tiles for 1,200 km2 = 15-60+ h on this VM,
  well under an hour batched on Broad's RTX 5070.

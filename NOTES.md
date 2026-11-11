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

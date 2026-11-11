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

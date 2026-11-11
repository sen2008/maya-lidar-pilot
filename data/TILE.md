# Tile manifest — pilot tile

## Tile

- **File:** `ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif` (NOT in this repo — 407 MB)
- **Size:** 426,482,130 bytes (verified via `Content-Range` on a range GET)
- **Product:** G-LiHT Digital Terrain Model, 0.5 m/px, ground-classified bare earth
- **CRS:** EPSG:32615 (UTM zone 15N)
- **Dimensions:** 13,872 x 7,686 px
- **Bounds (UTM 15N, m):** E 694,565.45 – 701,501.27, N 2,027,569.34 – 2,031,412.30
  (~6.94 km E–W x 3.84 km N–S, southwestern Campeche, Mexico)
- **Nodata:** 1.70141e38, dtype float32

## Source (no login needed)

- **Mirror used:** Scholar@UC (University of Cincinnati) collection
  "G-LiHT Yucatan Peninsula - 50cm / pixel DEM GeoTIFF files" (2026
  reprocessing of the original G-LiHT LAS files from the Goddard home
  archive, glihtdata.gsfc.nasa.gov)
- **Direct download (no auth, verified 2026-11-11):**
  https://scholar.uc.edu/downloads/vh53wx56r
  (HEAD returns HTTP 500 on this host; ranged GET works, returns 206.)
- **Collection page:**
  https://scholar.uc.edu/concern/datasets/qb98mh115
  (dataset "Campeche_NFI - 50cm/pixel DEM File, GeoTIFF")
- **Canonical source:** NASA G-LiHT via LP DAAC (GLDTMK.001,
  doi:10.5067/Community/GLIHT/GLDTMK.001). A free NASA Earthdata login
  is required there; the UC mirror is the open mirror used for this pilot.
  G-LiHT data were collected April–May 2013 (AMIGACarb campaign).

## Validation ground truth

- **Šprajc (2026) nested complex #1**, Ancient Mesoamerica
  - Coordinates: 18.3354° N, 91.1130° W (WGS84)
  - UTM 15N: E 699,410.8, N 2,028,326.8
  - Position in tile: 4,845 m E and 757 m N of the tile SW corner — inside.
  - Published as: Ruhl et al. (2018) site, detected on G-LiHT transect
    I2s502 (= this file's l2s502); "nested construction" of low,
    elongated platforms in roughly concentric rectangles — interpreted
    as a possible Maya marketplace. Low platforms are hard to spot in
    the field and easy to spot in LiDAR, which makes this a good
    validation target: if the pipeline cannot flag this, it cannot
    be trusted on anything subtler.
  - Backup in-tile check: the transect should also contain ordinary
    residential mounds; see candidate list.

## Why not Valeriana

Valeriana (18.9025° N, 89.3175° W, announced Oct 2024) was found in a
2013 *environmental* LiDAR survey by The Nature Conservancy, not in
G-LiHT data. It is not known to fall inside any G-LiHT transect, so it
cannot validate a G-LiHT pipeline. The Šprajc complex #1 is explicitly
tied to this transect in the literature, so it is the validation site.

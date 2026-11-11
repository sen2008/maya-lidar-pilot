# Maya LiDAR Pilot

Pilot: screening NASA G-LiHT LiDAR for unmapped Maya settlement features in the
Yucatan / Campeche lowlands, validated against known sites.

Inspired by Pietro Miotti's Nov 10, 2026 report (48 agents, 1,200 km² of G-LiHT
data, one hilltop flagged, INAH replied nothing is registered there — field check
pending, discovery unconfirmed). This repo tests whether the same pipeline works
at a one-tile scale, with a known site as ground truth.

## Method

1. Pick one small G-LiHT tile containing a known, documented Maya site.
2. Build a ground-classified DTM, hillshade, slope, and local-relief views.
3. Screen for rectilinear platforms, mounds, causeways, plaza-like flats.
4. Validate: the pipeline must flag the known site before any other candidate
   is taken seriously.
5. Cross-check candidates against the known-site literature.

## Status

Pilot in progress. Nothing here is a confirmed site. No contact with INAH;
this is desk work only until a candidate survives validation.

## Layout

- `scripts/` — processing and screening code
- `data/` — tile manifest (NOT the raw LiDAR; too large, fetch from ORNL DAAC)
- `outputs/` — hillshades, candidate close-ups, candidate list

## Data

NASA G-LiHT (Goddard LiDAR, Hyperspectral & Thermal Imager).
Canonical source is LP DAAC (GLDTMK.001), which needs a free NASA
Earthdata login. This pilot used the open Scholar@UC mirror instead
(no login): 50 cm DEMs at https://scholar.uc.edu — see
`data/TILE.md` for the exact tile and download URL.

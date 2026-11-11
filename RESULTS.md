# Q2000 validation — does the trained model find the known site?

**Verdict: YES, with a caution. The published Q2000 model detects Šprajc (2026)
complex #1 on the pilot tile at both input resolutions — but at modest
confidence, and the full-tile scan would miss the known site at any practical
operating threshold (≥0.5). The paper's F1 0.89 does not obviously replicate
here; the package's own validation AP (~0.62) is the number this run is
consistent with.**

- Known site: Šprajc (2026) nested complex #1, 18.3354 N, 91.1130 W
  (UTM 699410.8 E, 2028326.8 N), already confirmed visible on the pilot hillshade.
- Tile: `ChiapsCampeche_NFI_l2s502_DEM_0p5m_v1.tif`, 13,872 × 7,686 px at
  0.5 m/px, EPSG:32615, SW Campeche.
- Model: Q2000_ExRev09 (YOLOv3 / DarkNet53, 61.6M params, 1 class
  "Maya Structures", 416×416 RGB input, 9 anchors), CC-BY 4.0,
  Benjamin Britton / University of Cincinnati.
- Date: 2026-11-11. All inference on this VM's CPU (~1 s/chip).

## 1. Method

### RVT composite (paper Table 3, reproduced from the 0.5 m DEM)

Four layers, all stretched to Min/Max, built with the open-source Relief
Visualization Toolbox (RVT_py, github.com/EarthObservation/RVT_py):

| Layer | Blend | Transparency | Brightness | Contrast | Gamma |
|---|---|---|---|---|---|
| 1 Multi-hillshade | Normal (base) | — | 10 | 20 | 2 |
| 2 Slope | Difference | 33 | 10 | 20 | 3 |
| 3 Open-Positive | Overlay | 33 | −30 | 55 | 1 |
| 4 Skyview | Multiply | 33 | 1 | 30 | 0.5 |

**Ambiguities recorded (paper does not specify these):**
- RVT parameters not in the paper: multi-hillshade = 16 directions, sun
  elevation 35°; slope in degrees; SVF/openness = 16 directions, radius
  10 px, noise 0. (RVT_py defaults.)
- ArcGIS brightness/contrast/gamma math is undocumented. Assumed: stretch
  to [0,1], then gamma as v^(1/gamma) (gamma > 1 brightens), contrast
  (v−0.5)·(1+c/100)+0.5, brightness v+b/100, order gamma → contrast →
  brightness, clip [0,1]; blend with alpha = 1 − transparency/100.
- Multi-hillshade is the MEAN of 16 directional hillshades (computed one at
  a time to save memory).
- Nodata filled with the window/block mean before RVT (as in the pilot).
- Min/max stretch is per-window for the window runs, per-TILE (global) for
  the full-tile scan. This matters — see §5.
- Composite is single-channel grayscale replicated to 3-channel RGB
  (as ArcGIS exports it).

### Network and weights

YOLOv3 DarkNet53 built to match the state-dict key layout
(eriklindernoren/PyTorch-YOLOv3 lineage). **Strict load passes:
missing [] unexpected []** — architecture check done. Two cfg-route bugs
found and fixed during the build (module 15 takes module 12's output, not
13's; module 23 takes module 20's, not 21's) and the detection convs need a
`.conv` wrapper module to match checkpoint keys (`module_list.14.conv.weight`).

### Inference

Sliding window, 416×416 chips, stride 208, padding 104 (chip centers must
lie ≥104 px inside a chip; edge detections discarded). Demo settings from
the author's screenshot: confidence 0.1, NMS overlap 0.1. The paper says
NMS 0.5 — both were run; see §4. Detection coordinates are UTM (EPSG:32615)
plus lon/lat in the merged JSON/GeoJSON. Cross-chip duplicates merged by
global NMS on UTM boxes (the 0.61 and 0.13 boxes at the same spot in the
600 m site run were one feature in two chips).

## 2. Results — window runs (per-window stretch)

| Run | Chips | Raw detections (conf ≥ 0.1, NMS 0.1) | Merged | ≤150 m of site #1 | Max conf |
|---|---|---|---|---|---|
| Site 600 m, 0.5 m | 14 | 10 | 9 | 8 | 0.61 |
| Control 600 m, 0.5 m | 10 | 4 | 4 | 0 | 0.56 |
| Site 1200 m, 0.5 m | 38 | 24 | 23 | 6 | 0.93 |
| Site 1200 m, 1.0 m | 7 | 13 | 13 | 5 | 0.72 |
| Control 1200 m, 0.5 m | 37 | 17 | 17 | 6 | 0.93 |
| Control 1200 m, 1.0 m | 8 | 7 | 7 | 0 | 0.57 |

Note: the 1200 m "control" window overlaps the site window (centers only
~800 m apart, windows 1200 m wide), so it is NOT a clean control — it
re-detects the same features. The 600 m control is the valid comparison.

### Detections at the known site (site 1200 m, 0.5 m, merged, ≤150 m)

| Conf | UTM E | UTM N | Size (m) | Dist to site #1 (m) | Lon | Lat |
|---|---|---|---|---|---|---|
| 0.69 | 699416 | 2028265 | 52×51 | 62 | −91.11296 | 18.33539 |
| 0.69 | 699408 | 2028388 | 32×30 | 62 | −91.11303 | 18.33595 |
| 0.53 | 699671 | 2028284 | 37×37 | 264 | — | — |
| 0.41 | 699459 | 2028415 | 48×55 | 100 | — | — |
| 0.37 | 699500 | 2028319 | 48×49 | 89 | — | — |
| 0.30 | 699438 | 2028341 | 35×40 | 31 | — | — |

(600 m window, strongest: 0.61 @ 64 m, 50×48 m — same feature as the 0.69
box above; the 0.61/0.13 cross-chip duplicate was merged. Full lists with
lon/lat in `outputs/detections_merged.json` / `.geojson`.)

**The known complex is detected.** The strongest box (0.61–0.69, 50×48 m)
sits ~62–64 m from the published coords, on the square platform the pilot
hillshade showed at ~699300–699420 E, 2028260–2028470 N; several more boxes
(0.30–0.41) land on the ring cluster and the rectangular platform around it.
The complex is a cluster of structures, and the model fires on several of
them, not one clean box.

## 3. Full-tile scan (global stretch, 0.5 m)

Block-wise scan of the whole tile (2080 px blocks, 64 px halo, 13 blocks
with data, 215 chips, 180 s inference): **42 detections**.

- Confidence distribution: ≥0.5: 7; 0.3–0.5: 7; 0.1–0.3: 28.
- Within 150 m of site #1: 5 detections — top 0.37 @ 101 m, 0.29 @ 60 m,
  0.23 @ 135 m, 0.21 @ 114 m, 0.21 @ 54 m.
- Top detections anywhere on the tile:
  - 0.90 @ 699081 E, 2028377 N (333 m WNW of site #1)
  - 0.86 @ 698441 E, 2028526 N (990 m NW)
  - 0.81 @ 699064 E, 2028569 N (423 m NW)
  - 0.56 @ 696786 E, 2029145 N (2.75 km NW)
  - 0.53 @ 699882 E, 2028191 N (490 m SE)
- Detection density ≈ 42 over ~5–6 km² of valid swath ≈ 7–8 /km², far below
  the landscape's ~55 structures/km² — the model under-detects relative to
  structure density (or most structures sit below its threshold).

## 4. The 0.5 m vs 1 m question — 0.5 m wins

Same 1200 m window, same settings: 0.5 m gives 24 detections (max 0.93);
1 m gives 13 (max 0.72). 0.5 m finds ~2× as many and higher confidences.
Both detect the site (0.69 @ 62 m at 0.5 m; 0.72 @ 63 m + 0.39 @ 24 m at
1 m). **Use 0.5 m** — consistent with the .emd cell size of 0.513 m.

## 5. NMS 0.1 vs 0.5 — barely matters

Site 1200 m: NMS 0.1 → 24 detections (0.5 m) / 13 (1 m); NMS 0.5 → 28 / 14.
The paper's 0.5 keeps a few more boxes; the site detection is present at
both. (600 m window earlier: identical counts at both thresholds.)

## 6. Honest assessment

**What the model gets right:**
- It finds the known site at both resolutions, in the window runs and in
  the full-tile scan. The validation question — "does the published model
  actually find the known Maya site on our tile?" — is answered yes.
- It is stable, not random: the same features are detected independently
  in overlapping windows and in the scan (0.90/0.93 @ 699081, 2028377;
  0.86/0.78 @ 698441, 2028526; 0.81/0.86/0.60 @ 699064, 2028569).
- It is a much better filter than the pilot's crude threshold screen:
  23 detections vs 58 threshold components in the same 1200 m window,
  ranked by confidence, and it fires on the same structure cluster the
  crude screen hit (crude candidates #227 @ 29 m, #194 @ 57 m, #166 @ 62 m
  match Q2000 boxes at 31–64 m).

**What limits it:**
- **Confidence is modest.** Most detections are 0.1–0.3; only 7 of 42
  scan detections reach 0.5. The known site itself tops out at 0.61–0.69
  in the window runs and **0.37 in the full-tile scan — at any practical
  operating threshold (≥0.5) the scan would MISS the known site.**
- **Stretch sensitivity.** The site scores 0.69 with a per-window stretch
  but 0.37 with the per-tile (global) stretch. The model was presumably
  trained on per-raster composites, so the global stretch is the faithful
  one — and it is the weaker one. Any scale-up must fix the stretch
  convention (per-tile, per-transect, or per-window) and re-validate.
- **The control is not a true negative.** The 600 m control window still
  yields 4 detections (max 0.56), and its hillshade shows rectilinear
  features — this landscape is dense settlement, so "background" detections
  may be real unrecorded structures. No false-positive rate can be computed
  without ground truth for the whole tile, and no F1 can be computed either.
- **The F1 0.89 headline does not obviously replicate.** The package's own
  validation AP is ~0.62 (best epoch AP 0.6217, accuracy 0.6248), and this
  run — modest confidences, a long low-confidence tail, threshold-sensitive
  site detection — is consistent with ~0.62, not 0.89. Treat the model as a
  candidate generator that needs human review of every box, not as a
  detector that can be trusted at a fixed threshold.

**No new site is claimed.** The high-confidence detections away from site #1
(0.90 @ 699081, 2028377; 0.86 @ 698441, 2028526; 0.81 @ 699064, 2028569) are
unvetted candidates in a dense-settlement landscape. They are worth a
hillshade look, not a discovery claim.

## 7. What a scale-up would need

- **Stretch convention fixed and validated** (per-tile vs per-transect vs
  per-window — the site's confidence moves 0.37 → 0.69 with this choice).
- **A ground-truth set**: the published Šprajc (2026) complexes on this
  transect are the only labelled sites; more labelled positives and a
  genuinely negative control area are needed before any F1 claim.
- **Threshold calibration** on labelled data (the 0.1 demo threshold is a
  candidate-generation setting, not a detection threshold).
- **Compute**: this VM did one tile in ~5 min of RVT + 3 min of inference
  (215 chips). 1,200 km² is ~200 tiles at this density — 15–60+ hours on
  this VM. Broad's RTX 5070 would do it in well under an hour batched.
  The scan script (`scripts/scan_tile.py`) already tiles the work; the
  remaining piece is a driver over many tiles plus a cross-tile dedup.
- **Human review of every candidate** — the model ranks, it does not decide.

## Files

- `scripts/make_composite.py` — window RVT composite + chip export (Table 3).
- `scripts/scan_tile.py` — full-tile block scan (2080 px blocks, 64 px halo).
- `scripts/yolov3.py` — YOLOv3 DarkNet53 matching the state dict (strict load ✓).
- `scripts/infer.py` — sliding-window inference (conf/NMS flags).
- `scripts/merge_detections.py` — cross-chip global NMS + lon/lat + GeoJSON.
- `scripts/plot_detections.py` — detection figure.
- `outputs/detections_merged.json` / `.geojson` — window runs (site/control,
  0.5 m and 1 m, NMS 0.1; NMS 0.5 variants in `outputs/detections_*_nms05.json`).
- `outputs/detections_scan_merged.json` / `.geojson` — full-tile scan, 42 detections.
- `outputs/q2000_detections_figure.png` — site window + full tile, detections overlaid.
- `outputs/composite/` — window composites and chip manifests (site/control,
  600 m and 1200 m, 0.5 m and 1 m).
- `outputs/scan_0.5m/` — full-tile scan manifest (`chips.json`); chips and
  per-block RVT layers are local-only intermediates (regenerable via
  `scan_tile.py`), not pushed to the repo.
- Model files (`.dlpk`, `.pth`) and the DEM are NOT in the repo.

# Q2000 Maya-detection model — inspection notes

Trained YOLOv3 model for detecting Maya structures in G-LiHT LiDAR, from
Britton et al., *Journal of Archaeological Method and Theory* 2026
("Evaluating Broadscale Deep Learning for Maya Settlement Detection in
G-LiHT Lidar", doi:10.1007/s10816-025-09741-5).

## Source

- Dataset page: https://scholar.uc.edu/concern/datasets/3f462716x
- DOI: 10.7945/tbvg-8n31
- Creator: Benjamin Britton, University of Cincinnati
- **License: CC-BY 4.0** (per the dataset page). Attribution required.
  Note: this is *not* public domain, despite some summaries saying so.
- Paper PDF also on the dataset page:
  `EvaluatingBroadscale_082325_Britton.pdf`

## Download

- Direct URL: https://scholar.uc.edu/downloads/79407z82r
- File: `Q2000_ExRev09.dlpk` (note: the file is named ExRev09, not `Q2000.dlpk`)
- Size: 230,312,157 bytes (~219.6 MiB) — matches the server Content-Range
- SHA-256: `ca0e697aafebcfec61abd9d078738a85dbb768278924cc5a37417982e6b398e2`
- No login needed. The canonical LP DAAC route needs a free Earthdata login,
  but Scholar@UC serves the file openly. Host quirk: HEAD returns 500;
  ranged GET works; plain curl works through the egress proxy (pip's 15 s
  default timeout does NOT — use curl for large files here).

## Unpacked inventory (`.dlpk` is a zip)

| File | Size | Notes |
|---|---|---|
| `Q2000_ExRev09.pth` | 246,404,961 B (~235 MiB) | weights |
| `Q2000_ExRev09.emd` | 2,232 B | model definition (JSON, read in full) |
| `model_metrics.html` | 933,168 B | boilerplate; only metric is accuracy 0.6248 |
| `ModelCharacteristics/training_validation_loss.json` | 2,126 B | 9 epochs of loss/AP |
| `ModelCharacteristics/loss_graph.png` | 33,997 B | |
| `ModelCharacteristics/show_results.png` | 665,381 B | |

Local copies: `models/Q2000_ExRev09.dlpk`, `models/Q2000_unpacked/`.
The `.dlpk` / `.pth` are **not** in the git repo (too large); this file is
the record. Re-download from the URL + checksum above.

## Model definition (`Q2000_ExRev09.emd`, full read)

- YOLOv3, backbone DarkNet53, backend PyTorch, ArcGIS Learn 2.3.0
- 1 class: `Maya Structures`
- Input: 416 x 416, 3 bands (RGB), NCHW
- Anchors (9, px at 416 input):
  `[15,19] [67,24] [32,86] [71,71] [143,63] [106,102] [88,164] [161,149] [215,242]`
- Min/MaxCellSize: 0.513 m (EPSG:32615) — see the resolution note below
- Recorded accuracy: `{"Maya Structures": 0.6248414717331935}`
- `monitored_valid_loss`: 103.78

## Weights — verified extractable

`Q2000_ExRev09.pth` is a **legacy pickle-protocol-2** `torch.save`
(not the zip format). Verified by loading with torch 2.14.1+cpu
(`torch.load(..., weights_only=True)` — loads cleanly):

- Plain `OrderedDict`, **438 tensors, 61.6M params**, float32
  (+ int64 `num_batches_tracked`)
- Key layout `module_list.N(.module_list.M.I).conv / batch_norm` — the
  classic Darknet/YOLOv3 lineage (ArcGIS's implementation derives from it)
- First conv: `(32, 3, 3, 3)`; weight stats sane (mean -0.007, std 0.38)
- Detection heads at `module_list.14 / 22 / 28`, all with 18 output
  channels = 3 anchors x (4 bbox + 1 objectness + 1 class) — confirms
  **1 class**. Head input channels 1024 / 512 / 256 (13x13 / 26x26 /
  52x52 grids, standard YOLOv3 three-scale).
- Top-level indices 0–28 with gaps at 16 and 24: the parameter-free
  route/upsample layers. No `yolo`/`route`/`shortcut` parameter keys.
- **Verdict: the weights are a standard YOLOv3 state dict and load
  directly into an open Darknet implementation with matching key names.
  No ArcGIS needed to read them.**

## Training metrics (from the package, not the paper)

- 9 epochs; validation AP ~0.58–0.62, best epoch 1 (AP 0.6217);
  validation loss ~104–109 (ArcGIS's loss scale, not comparable to
  standard YOLO loss).
- Paper reports test **F1 0.89** for Q2000. That and the package's
  AP ~0.62 are different metrics (test-set F1 at a chosen threshold vs
  validation AP); the real check is running it on our pilot tile against
  the known site.

## Input recipe — the important part

The model was **not** trained on a plain hillshade. From the paper, the
input is a 4-layer Relief Visualization Toolkit (RVT) composite built
from the DEM: **MultiHillshade + Slope + Open-Positive + Skyview**,
layered with the paper's Table settings (transparency 33% each on the
top three layers, blend types Multiply/Overlay/Difference/Normal,
per-layer brightness/contrast/gamma). RVT is open source, so the recipe
is reproducible without ArcGIS — but the exact table values must be
lifted from the paper, and the composite must be reproduced faithfully
or the model sees a different distribution than it trained on.

Resolution: the paper used NASA's published **1 m/px** DTMs. The `.emd`
cell size is **0.513 m**, matching the 0.5 m reprocessed DEMs (our pilot
tile is 0.5 m/px). The ExRev09 revision may therefore expect ~0.5 m
input. **Test both**: run the model on the pilot tile at 0.5 m/px and
at 1 m/px resampled, and compare detections against the known
Šprajc complex #1.

Demo parameters (from `Q2000DemoScreenCap.JPG`, the author's own ArcGIS
Pro run — the most direct evidence of inference settings):

- confidence **threshold 0.1** (very low; detections shown at 10–60%)
- **nms_overlap 0.1**, padding 104, batch_size 4,
  exclude_pad_detections True, test_time_augmentation False
- Note: the paper mentions NMS ratio 0.5; the demo uses 0.1. Start
  with the demo's 0.1 / 0.1 and tune against the known site.

## Conversion / run plan (no ArcGIS Pro)

1. **Reproduce the input composite.** From our 0.5 m DEM, generate the
   4 RVT layers (MultiHillshade, Slope, Open-Positive, Skyview) with
   RVT itself or a Python reimplementation, composited per the paper's
   table. Export 3-band RGB.
2. **Build the network.** A Darknet-YOLOv3 implementation matching the
   state-dict key layout (the eriklindernoren/PyTorch-YOLOv3 lineage
   the ArcGIS model derives from). Load the state dict with a strict
   load — any key mismatch fails loudly, which is the check that the
   architecture matches.
3. **Tile and infer.** Sliding 416x416 windows over the composite
   (padding 104, overlap to avoid edge misses), confidence 0.1,
   NMS IoU 0.1, map boxes back to UTM, write GeoJSON.
4. **Validate on the pilot tile.** Must detect the known Šprajc
   complex #1. A/B 0.5 m vs 1 m inputs; tune the confidence threshold
   until the known site is found without the control window lighting up.
5. **Scale.** CPU is fine for one tile (the pilot window is ~100–200
   chips). Full 1,200 km² is ~7k chips at 1 m/px (~28k at 0.5 m/px,
   more with overlap) — that wants a GPU. Broad's RTX 5070 (via SSH)
   would run the whole scan in well under an hour batched; on this VM
   it would be 15–60+ hours. Do not use Broad yet; decide at scale-up.

## Unknowns

- Exact RVT table values (in the paper; need a careful read of the table).
- Whether ExRev09 expects 0.5 m or 1 m input (A/B test settles it).
- ArcGIS chip preprocessing details (normalization to [0,1] assumed;
  any mean/std standardization is undocumented — check by testing).
- Confidence/NMS discrepancy between paper (0.5) and demo (0.1).
- Whether the paper's F1 0.89 replicates on our tile (the package's
  own AP ~0.62 suggests real-world performance is lower).

## Compute notes

- Scratch torch env: `models/venv-torch/` (torch 2.14.1+cpu, installed
  from a curl-downloaded wheel; pip direct download times out through
  the proxy). Keep it out of the repo.
- Loading the 246 MB checkpoint on this VM works (needs ~1 GB RAM
  briefly); CPU inference per 416-chip is seconds at most.
- GPU: not needed for the pilot. Broad RTX 5070 via SSH if/when the
  full scan happens.

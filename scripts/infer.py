#!/usr/bin/env python3
"""Run Q2000 (YOLOv3) on composite chips and write detections.

Usage (torch venv: models/venv-torch/bin/python):
  infer.py --manifest outputs/composite/site_0.5m/chips.json \
           --chips-dir outputs/composite/site_0.5m/chips \
           --out outputs/detections_site_0.5m.json \
           --conf 0.1 --nms 0.1

Settings (from the author's demo screenshot Q2000DemoScreenCap.JPG):
  confidence 0.1, nms_overlap 0.1, padding 104 (chips overlap by 104 px on
  each side; detections whose box centre falls in the padded margin are
  excluded), batch 4, no test-time augmentation.
  The paper says NMS 0.5 — run again with --nms 0.5 to compare.

Input chips: 416x416 uint8 RGB .npy, values 0-255; normalized to [0,1]
(the ArcGIS default chip preprocessing; no mean/std in the .emd).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolov3 import YOLOv3, decode_head, nms, ANCHORS, MASKS

ROOT = os.path.expanduser("~/workspace/lidar-pilot")
WEIGHTS = os.path.join(ROOT, "models/Q2000_unpacked/Q2000_ExRev09.pth")
SITE_E, SITE_N = 699410.8, 2028326.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--nms", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(4)
    model = YOLOv3()
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"strict-load check: missing={missing} unexpected={unexpected}")
    assert not missing and not unexpected, "state dict does not match network"
    model.eval()
    print("weights strict-loaded: architecture matches the checkpoint")

    with open(args.manifest) as fh:
        man = json.load(fh)
    chips = man["chips"]
    cell = man["cell"]
    print(f"{len(chips)} chips at {cell} m/px, conf>={args.conf}, NMS {args.nms}")

    all_dets = []
    t0 = time.time()
    for i in range(0, len(chips), args.batch):
        batch = chips[i:i + args.batch]
        imgs = []
        for c in batch:
            arr = np.load(os.path.join(args.chips_dir, c["id"] + ".npy"))
            imgs.append(torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0))
        x = torch.stack(imgs)
        with torch.no_grad():
            y1, y2, y3 = model(x)
        for j, c in enumerate(batch):
            boxes = []
            for raw, mask in zip((y1, y2, y3), MASKS):
                boxes.append(decode_head(raw[j:j + 1],
                                        [ANCHORS[k] for k in mask])[0])
            boxes = torch.cat(boxes, dim=0)
            raw_n = int((boxes[:, 4] >= args.conf).sum())
            keep = boxes[boxes[:, 4] >= args.conf]
            # exclude detections centred in the 104 px pad margin
            if len(keep):
                cx = (keep[:, 0] + keep[:, 2]) / 2
                cy = (keep[:, 1] + keep[:, 3]) / 2
                central = (cx >= 104) & (cx < 312) & (cy >= 104) & (cy < 312)
                keep = keep[central]
            kept = nms(keep, args.nms) if len(keep) else keep
            for b in kept:
                x1, y1_, x2, y2_, conf, cls = [float(v) for v in b]
                cx_px, cy_px = (x1 + x2) / 2, (y1_ + y2_) / 2
                w_m, h_m = (x2 - x1) * cell, (y2_ - y1_) * cell
                e_c = c["e_left"] + cx_px * cell
                n_c = c["n_top"] - cy_px * cell
                all_dets.append({
                    "chip": c["id"], "tag": c["tag"], "res_m": cell,
                    "conf": round(conf, 3),
                    "x1_px": round(x1, 1), "y1_px": round(y1_, 1),
                    "x2_px": round(x2, 1), "y2_px": round(y2_, 1),
                    "w_m": round(w_m, 1), "h_m": round(h_m, 1),
                    "e": round(e_c, 1), "n": round(n_c, 1),
                    "dist_site_m": round(float(np.hypot(e_c - SITE_E, n_c - SITE_N)), 1),
                })
            print(f"  {c['id']}: raw>={args.conf}: {raw_n}, after pad-excl+NMS: {len(kept)}")
    print(f"inference done in {time.time()-t0:.0f}s: {len(all_dets)} detections")
    with open(args.out, "w") as fh:
        json.dump({"settings": {"conf": args.conf, "nms": args.nms,
                                "pad_exclude_px": 104, "chip_size": 416,
                                "stride": man["stride"], "cell_m": cell,
                                "weights": "Q2000_ExRev09.pth"},
                   "n_chips": len(chips), "n_detections": len(all_dets),
                   "detections": all_dets}, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

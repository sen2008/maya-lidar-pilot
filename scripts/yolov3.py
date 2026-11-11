#!/usr/bin/env python3
"""YOLOv3 (DarkNet53) matching the Q2000 state dict key layout exactly.

Key layout (from Q2000_ExRev09.pth, 438 tensors, 61.6M params):
  module_list.0   conv 32  (3,3,3) s1
  module_list.1   conv 64  s2
  module_list.2   residual x1 (64ch):  [conv 32 1x1, conv 64 3x3] + shortcut
  module_list.3   conv 128 s2
  module_list.4   residual x2 (128ch)
  module_list.5   conv 256 s2
  module_list.6   residual x8 (256ch)
  module_list.7   conv 512 s2
  module_list.8   residual x8 (512ch)
  module_list.9   conv 1024 s2
  module_list.10  residual x4 (1024ch)
  module_list.11  sequential 2x [conv 512 1x1, conv 1024 3x3]
  module_list.12  conv 512 1x1
  module_list.13  conv 1024 3x3
  module_list.14  conv 18 1x1  (YOLO head, 13x13)
  module_list.15  conv 256 1x1
  module_list.16  (upsample, no params)
  module_list.17  conv 256 1x1 (768 in = 256 upsampled + 512 route from module 8)
  module_list.18  conv 512 3x3
  module_list.19  sequential 1x [conv 256 1x1, conv 512 3x3]
  module_list.20  conv 256 1x1
  module_list.21  conv 512 3x3
  module_list.22  conv 18 1x1  (YOLO head, 26x26)
  module_list.23  conv 128 1x1
  module_list.24  (upsample, no params)
  module_list.25  conv 128 1x1 (384 in = 128 upsampled + 256 route from module 6)
  module_list.26  conv 256 3x3
  module_list.27  sequential 2x [conv 128 1x1, conv 256 3x3]
  module_list.28  conv 18 1x1  (YOLO head, 52x52)

This is the standard yolov3.cfg layer order; residual units are grouped
into single modules (module_list.N.module_list.K.0/1) and the YOLO head
conv pairs are grouped the same way (modules 11/19/27, sequential, no
shortcut). Detection convs (14/22/28) have bias, no BN, no activation.
Anchors from the .emd (px at 416 input), masks [[6,7,8],[3,4,5],[0,1,2]]
so the 13x13 head gets the 3 largest anchors, 52x52 the 3 smallest.
"""
import torch
import torch.nn as nn


ANCHORS = [(15, 19), (67, 24), (32, 86), (71, 71), (143, 63),
           (106, 102), (88, 164), (161, 149), (215, 242)]
MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # heads: 13x13, 26x26, 52x52


class ConvBNLeaky(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride, k // 2 if k == 3 else 0,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))


class ConvPair(nn.Module):
    """One sequential [1x1, 3x3] conv pair. Keys: module_list.N.module_list.K.0/1."""
    def __init__(self, ch_in, ch_mid, ch_out):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(ConvBNLeaky(ch_in, ch_mid, 1),
                          ConvBNLeaky(ch_mid, ch_out, 3)),
        ])

    def forward(self, x):
        return self.module_list[0](x)


class ResidualBlock(nn.Module):
    """One residual unit: x + conv3x3(conv1x1(x)). Keys as ConvPair."""
    def __init__(self, ch):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(ConvBNLeaky(ch, ch // 2, 1),
                          ConvBNLeaky(ch // 2, ch, 3)),
        ])

    def forward(self, x):
        return x + self.module_list[0](x)


class ResidualStack(nn.Module):
    """n residual units in sequence. Keys module_list.N.module_list.K.0/1."""
    def __init__(self, ch, n):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(ConvBNLeaky(ch, ch // 2, 1),
                          ConvBNLeaky(ch // 2, ch, 3))
            for _ in range(n)
        ])

    def forward(self, x):
        for unit in self.module_list:
            x = x + unit(x)
        return x


class SequentialPairs(nn.Module):
    """n sequential [1x1, 3x3] pairs (YOLO head sections, no shortcut)."""
    def __init__(self, pairs):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(ConvBNLeaky(a, b, 1), ConvBNLeaky(b, c, 3))
            for (a, b, c) in pairs
        ])

    def forward(self, x):
        for unit in self.module_list:
            x = unit(x)
        return x


class YoloHead(nn.Module):
    """Detection conv. Keys: module_list.N.conv.weight/bias (wrapped in .conv)."""
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, 18, 1, bias=True)

    def forward(self, x):
        return self.conv(x)


class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        m = []
        m.append(ConvBNLeaky(3, 32, 3, 1))          # 0
        m.append(ConvBNLeaky(32, 64, 3, 2))         # 1
        m.append(ResidualStack(64, 1))              # 2
        m.append(ConvBNLeaky(64, 128, 3, 2))        # 3
        m.append(ResidualStack(128, 2))             # 4
        m.append(ConvBNLeaky(128, 256, 3, 2))       # 5
        m.append(ResidualStack(256, 8))             # 6
        m.append(ConvBNLeaky(256, 512, 3, 2))       # 7
        m.append(ResidualStack(512, 8))             # 8
        m.append(ConvBNLeaky(512, 1024, 3, 2))      # 9
        m.append(ResidualStack(1024, 4))            # 10
        m.append(SequentialPairs([(1024, 512, 1024), (1024, 512, 1024)]))  # 11
        m.append(ConvBNLeaky(1024, 512, 1))         # 12
        m.append(ConvBNLeaky(512, 1024, 3))         # 13
        yolo1 = YoloHead(1024)  # 14
        m.append(yolo1)
        m.append(ConvBNLeaky(512, 256, 1))          # 15
        m.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 16
        m.append(ConvBNLeaky(768, 256, 1))          # 17
        m.append(ConvBNLeaky(256, 512, 3))          # 18
        m.append(SequentialPairs([(512, 256, 512)]))  # 19
        m.append(ConvBNLeaky(512, 256, 1))          # 20
        m.append(ConvBNLeaky(256, 512, 3))          # 21
        yolo2 = YoloHead(512)  # 22
        m.append(yolo2)
        m.append(ConvBNLeaky(256, 128, 1))          # 23
        m.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 24
        m.append(ConvBNLeaky(384, 128, 1))          # 25
        m.append(ConvBNLeaky(128, 256, 3))          # 26
        m.append(SequentialPairs([(256, 128, 256), (256, 128, 256)]))  # 27
        yolo3 = YoloHead(256)  # 28
        m.append(yolo3)
        self.module_list = nn.ModuleList(m)

    def forward(self, x):
        ml = self.module_list
        x = ml[0](x)
        x = ml[1](x)
        x = ml[2](x)
        x = ml[3](x)
        x = ml[4](x)
        x = ml[5](x)
        x6 = ml[6](x)          # 256ch, 52x52 — route for head 3
        x = ml[7](x6)
        x8 = ml[8](x)
        x = ml[9](x8)
        x = ml[10](x)
        x = ml[11](x)
        x12 = ml[12](x)        # 512ch — route for module 15 (cfg layer 79)
        x = ml[13](x12)
        y1 = ml[14](x)
        x = ml[15](x12)
        x = ml[16](x)
        x = torch.cat([x, x8], dim=1)
        x = ml[17](x)
        x = ml[18](x)
        x = ml[19](x)
        x20 = ml[20](x)        # 256ch — route for module 23 (cfg layer 89)
        x = ml[21](x20)
        y2 = ml[22](x)
        x = ml[23](x20)
        x = ml[24](x)          # upsample 26->52
        x = torch.cat([x, x6], dim=1)
        x = ml[25](x)
        x = ml[26](x)
        x = ml[27](x)
        y3 = ml[28](x)         # 52x52
        return y1, y2, y3


def decode_head(raw, anchors, img_size=416):
    """raw: (B, 18, G, G) -> boxes (B, G*G*3, 6): x1,y1,x2,y2,conf,cls_conf."""
    B, _, G, _ = raw.shape
    raw = raw.view(B, 3, 6, G, G).permute(0, 1, 3, 4, 2).contiguous()
    tx = torch.sigmoid(raw[..., 0])
    ty = torch.sigmoid(raw[..., 1])
    tw = raw[..., 2]
    th = raw[..., 3]
    obj = torch.sigmoid(raw[..., 4])
    cls = torch.sigmoid(raw[..., 5])
    gy, gx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    gx = gx.to(raw.device).float()
    gy = gy.to(raw.device).float()
    bx = (tx + gx) / G * img_size
    by = (ty + gy) / G * img_size
    aw = torch.tensor([a[0] for a in anchors], dtype=torch.float32).view(1, 3, 1, 1)
    ah = torch.tensor([a[1] for a in anchors], dtype=torch.float32).view(1, 3, 1, 1)
    bw = aw * torch.exp(tw.clamp(-10, 10))
    bh = ah * torch.exp(th.clamp(-10, 10))
    conf = obj * cls
    x1 = (bx - bw / 2).reshape(B, -1)
    y1 = (by - bh / 2).reshape(B, -1)
    x2 = (bx + bw / 2).reshape(B, -1)
    y2 = (by + bh / 2).reshape(B, -1)
    return torch.stack([x1, y1, x2, y2, conf.reshape(B, -1),
                        cls.reshape(B, -1)], dim=-1)


def nms(boxes, iou_thresh):
    """boxes: (N, 6) x1,y1,x2,y2,conf,cls. Classic greedy NMS, single class."""
    if len(boxes) == 0:
        return boxes
    order = boxes[:, 4].argsort(descending=True)
    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return boxes[keep]

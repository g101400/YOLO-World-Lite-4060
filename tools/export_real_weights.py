#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方 yolov8s-worldv2.pt → ncnn (v3: 文本嵌入作为网络第二输入)

架构事实:
  - 骨干 4x C2fAttn 以 txt_feats 为 guide  → 视觉特征是文本条件化的
  - WorldDetect 头: cv4(BNContrastiveHead) 用原始 txt 打分
  → 文本必须作为网络输入; 最多 K=10 个提示词类(固定形状), 不足由端侧循环填充

输出约定:
  output0 = boxes [1,4,A]   xyxy, letterbox 640 坐标
  output1 = scores [1,10,A] 已 sigmoid(0=填充类), 端侧只取前 num_classes 行
输入约定:
  images [1,3,640,640] RGB/255
  txt    [1,10,512]    CLIP 文本嵌入(图内会再归一化)
"""
import os, sys, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = 'E:/MY_3d/YOLO-World-Lite-v1.0/android_demo/ncnn-android-yoloworld'
os.chdir(ROOT)

W = os.path.join(ROOT, '_real/yolov8s-worldv2.pt')
PNNX = os.path.join(ROOT, '_real/pnnx-20260704-windows/pnnx.exe')
WORK = os.path.join(ROOT, '_real/export')
IMG = os.path.join(ROOT, '_real/bus.jpg')
os.makedirs(WORK, exist_ok=True)

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2fAttn
from ultralytics.nn.modules.head import WorldDetect
from ultralytics.utils.tal import dist2bbox, make_anchors

REG_MAX = 16
K = 10

class ExportWorld(nn.Module):
    def __init__(self, ymodel):
        super().__init__()
        seq = ymodel.model.model
        self.body = seq[:-1]
        head = seq[-1]
        self.head_f = head.f                 # [15,18,21]
        self.cv2 = head.cv2
        self.cv3 = head.cv3
        self.cv4 = head.cv4                  # BNContrastiveHead per level (BN+scale+bias 全在图内)
        self.dfl = head.dfl
        self.strides = head.stride
        self.nl = len(head.cv2)
        # 预计算锚点(常量 buffer, 避免图内 meshgrid 产生 pnnx.Expression)
        with torch.no_grad():
            h = [640 // int(s) for s in self.strides.flatten().tolist()]
            fake = [torch.zeros(1, 1, hh, hh) for hh in h]
            ap, st = make_anchors(fake, self.strides, 0.5)
        self.register_buffer('anchor_points', ap)          # [A,2]
        self.register_buffer('stride_tensor', st)          # [A,1]

    def forward(self, images, txt):
        x = images
        ys = []
        for m in self.body:
            if m.f != -1:
                x = ys[m.f] if isinstance(m.f, int) else [x if j == -1 else ys[j] for j in m.f]
            if isinstance(m, C2fAttn):
                x = m(x, txt)
            else:
                x = m(x)
            ys.append(x)
        feats = [ys[j] for j in self.head_f]

        box = torch.cat([self.cv2[i](feats[i]).view(1, REG_MAX * 4, -1) for i in range(self.nl)], 2)
        scores = torch.cat([self.cv4[i](self.cv3[i](feats[i]), txt).view(1, K, -1)
                            for i in range(self.nl)], 2)
        dist = self.dfl(box)
        boxes = dist2bbox(dist, self.anchor_points.t().unsqueeze(0), xywh=False, dim=1) * self.stride_tensor.t()
        return boxes, scores.sigmoid()

def letterbox_rgb(img, size=640):
    from PIL import Image
    im = Image.open(img).convert('RGB')
    w0, h0 = im.size
    s = size / max(w0, h0)
    im = im.resize((int(w0 * s + 0.5), int(h0 * s + 0.5)))
    canvas = np.full((size, size, 3), 114, np.uint8)
    a = np.array(im); canvas[:a.shape[0], :a.shape[1]] = a
    return canvas

def main():
    print('== load & trace ==')
    y = YOLO(W); y.model.eval()
    wrapper = ExportWorld(y).eval()
    with torch.no_grad():
        img_d = torch.rand(1, 3, 640, 640)
        txt_d = F.normalize(torch.randn(1, K, 512), dim=-1)
        traced = torch.jit.trace(wrapper, (img_d, txt_d))
        traced.save(os.path.join(WORK, 'yoloworld.ts.pt'))
        tb, ts = wrapper(img_d, txt_d)
        print('trace OK: boxes', tuple(tb.shape), 'scores', tuple(ts.shape))

    print('== cross-validate vs ultralytics (3 classes padded to 10) ==')
    names = ['person', 'car', 'dog']
    y.set_classes(names)
    txt3 = y.model.txt_feats.detach()[0].clone()             # [3,512] unit
    # 端侧填充策略: 循环复制真实嵌入
    idx = torch.arange(K) % txt3.shape[0]
    txt10 = txt3[idx].unsqueeze(0)                           # [1,10,512]
    r = y.predict(IMG, conf=1e-6, iou=0.45, verbose=False)[0]
    bx_u = r.boxes.xyxy.numpy()
    cf_u = r.boxes.conf.numpy()
    cl_u = r.boxes.cls.numpy().astype(int)
    s = 640.0 / 1080.0   # bus.jpg 810x1080

    canvas = letterbox_rgb(IMG)
    x = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    with torch.no_grad():
        bx, sc = wrapper(x, txt10)
    sc = sc[0][:len(names)]                                  # [3,A]
    conf_a, cls_a = sc.max(dim=0)
    sel = (conf_a > 0.3).nonzero().flatten()
    print('wrapper anchors(conf>0.3):', len(sel))
    cand = torch.cat([bx[0].t()[sel], conf_a[sel, None], cls_a[sel, None].float()], 1).numpy()
    cand = cand[np.argsort(-cand[:, 4])][:8]
    for cx, cy, x2, y2, cf, cl in cand[:5]:
        print('   wrapper %s %.5f xyxy=%s' % (names[int(cl)], cf, [round(cx,1),round(cy,1),round(x2,1),round(y2,1)]))
    print('ultra top5:')
    for i in np.argsort(-cf_u)[:5]:
        print('   ultra   %s %.5f xyxy=%s' % (names[cl_u[i]], cf_u[i], np.round(bx_u[i]*s,1).tolist()))

    print('== pnnx ==')
    r2 = subprocess.run([PNNX, os.path.join(WORK, 'yoloworld.ts.pt'),
                         'inputshape=[1,3,640,640],[1,10,512]'],
                        cwd=WORK, capture_output=True, text=True, timeout=3600)
    print((r2.stdout or '')[-1000:])
    if r2.returncode != 0:
        print('STDERR:', (r2.stderr or '')[-2500:]); sys.exit(1)
    for f in sorted(os.listdir(WORK)):
        print('   workdir:', f)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pnnx 产物 → assets（blob 重命名）+ Python ncnn 运行时验证"""
import io
import os
import shutil
import sys

import numpy as np

ROOT = 'E:/MY_3d/YOLO-World-Lite-v1.0/android_demo/ncnn-android-yoloworld'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

RENAME = {'in0': 'images', 'in1': 'txt', 'out0': 'output0', 'out1': 'output1'}
SRC_PARAM = '_real/export/yoloworld.ts.ncnn.param'
SRC_BIN = '_real/export/yoloworld.ts.ncnn.bin'
DST_PARAM = 'app/src/main/assets/yoloworld.param'
DST_BIN = 'app/src/main/assets/yoloworld.bin'


def rename_param():
    lines = io.open(SRC_PARAM, encoding='utf-8').read().splitlines()
    out = []
    for idx, l in enumerate(lines):
        t = l.split()
        if idx < 2 or not t:  # 头两行: magic + 层/blob 计数
            out.append(l)
            continue
        ni, no = int(t[2]), int(t[3])
        t[1] = RENAME.get(t[1], t[1])  # 层名同步改(仅 Input 层叫 in0/in1)
        ins = [RENAME.get(b, b) for b in t[4:4 + ni]]
        outs = [RENAME.get(b, b) for b in t[4 + ni:4 + ni + no]]
        rest = t[4 + ni + no:]
        out.append(' '.join(t[:4] + ins + outs + rest))
    io.open(DST_PARAM, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    shutil.copy2(SRC_BIN, DST_BIN)
    chk = io.open(DST_PARAM, encoding='utf-8').read()
    for old in RENAME:
        assert (' %s ' % old) not in chk, 'rename incomplete: ' + old
    print('renamed ok ->', DST_PARAM, os.path.getsize(DST_BIN), 'bytes bin')


def verify():
    import torch
    import ncnn
    from export_real_weights import ExportWorld, letterbox_rgb, K
    from ultralytics import YOLO

    y = YOLO('_real/yolov8s-worldv2.pt')
    y.model.eval()
    names = ['person', 'car', 'dog']
    y.set_classes(names)
    txt3 = y.model.txt_feats.detach()[0].clone()
    txt10 = txt3[torch.arange(K) % 3].unsqueeze(0)

    net = ncnn.Net()
    print('step1: load_param...', flush=True)
    assert net.load_param(DST_PARAM) == 0, 'load_param failed'
    print('step2: load_model...', flush=True)
    assert net.load_model(DST_BIN) == 0, 'load_model failed'
    print('ncnn load OK', flush=True)
    canvas = letterbox_rgb('_real/bus.jpg')
    arr = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    ex = net.create_extractor()
    print('step3: input images...', flush=True)
    r1 = ex.input('images', ncnn.Mat(np.ascontiguousarray(arr)))
    print('step4: input txt...', flush=True)
    r2 = ex.input('txt', ncnn.Mat(np.ascontiguousarray(txt10[0].numpy())))
    print('step5: extract...', r1, r2, flush=True)
    _, ob = ex.extract('output0')
    _, os_ = ex.extract('output1')
    print('out0 w,h,c =', ob.w, ob.h, ob.c, '| out1 w,h,c =', os_.w, os_.h, os_.c, flush=True)
    bn = ob.numpy()
    sn = os_.numpy()
    A = ob.w * ob.h // 4
    scores = sn.reshape(-1, A)
    boxes = bn.reshape(4, A)
    conf = scores[:3].max(axis=0)
    print('A =', A, '| conf>0.3 anchors:', int((conf > 0.3).sum()))
    for a in np.argsort(-conf)[:5]:
        k = int(scores[:3, a].argmax())
        print('   %s %.4f xyxy=%s' % (names[k], conf[a], np.round(boxes[:, a], 1)))


if __name__ == '__main__':
    rename_param()
    verify()

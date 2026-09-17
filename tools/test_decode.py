#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线验证 YOLO-World 解码算法（与 app/src/main/jni/yoloworld.cpp 逻辑一致）。

不依赖模型权重：用合成的 boxes / cls 嵌入 + 提示词嵌入，验证
  1) 文本-视觉相似度打分 + sigmoid
  2) 阈值过滤
  3) NMS
  4) 布局自动检测（[A,4]/[4,A]、[A,E]/[E,A] 都能正确解析）
  5) 去 letterbox 还原到原图坐标

运行: python3 tools/test_decode.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def nms(objects, thr):
    # objects: list of (x0,y0,x1,y1,label,prob)
    def area(o):
        return max(0.0, o[2] - o[0]) * max(0.0, o[3] - o[1])

    order = sorted(range(len(objects)), key=lambda i: objects[i][5], reverse=True)
    picked = []
    while order:
        i = order.pop(0)
        picked.append(i)
        order = [j for j in order if _iou(objects[i], objects[j]) <= thr]
    return picked


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[2], a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    uni = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / uni if uni > 0 else 0.0


def decode(box_tensor, cls_tensor, text_emb, names, target_size, img_w, img_h,
           temperature=0.07, prob_thr=0.30, nms_thr=0.45, box_format=0):
    """box_tensor / cls_tensor: 2D lists 形状由函数自动识别（见 yoloworld.cpp）。"""
    # 识别 A (anchors) 与 E (embed dim)
    total_box = len(box_tensor) * len(box_tensor[0])
    A = total_box // 4
    total_cls = len(cls_tensor) * len(cls_tensor[0])
    E = total_cls // A

    # 判断布局: 若 box 第一维长度 == A 则是 [A,4]，否则 [4,A]
    box_anchor_first = (len(box_tensor) == A)
    cls_anchor_first = (len(cls_tensor) == A)

    def get_box(a):
        if box_anchor_first:
            return box_tensor[a][:]
        else:
            return [box_tensor[k][a] for k in range(4)]

    def get_feat(a):
        if cls_anchor_first:
            return cls_tensor[a][:]
        else:
            return [cls_tensor[k][a] for k in range(E)]

    num_classes = len(names)
    # letterbox 还原
    if img_w > img_h:
        scale = target_size / img_w
        w = target_size
        h = int(img_h * scale)
    else:
        scale = target_size / img_h
        h = target_size
        w = int(img_w * scale)
    wpad = target_size - w
    hpad = target_size - h  # 仅右/下补边

    proposals = []
    for a in range(A):
        b = get_box(a)
        f = get_feat(a)
        if box_format == 0:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        else:
            x0, y0, x1, y1 = b[0] - b[2] * 0.5, b[1] - b[3] * 0.5, b[0] + b[2] * 0.5, b[1] + b[3] * 0.5

        best_c, best_logit = 0, -1e30
        for c in range(num_classes):
            te = text_emb[c * E:(c + 1) * E]
            s = dot(f, te) / temperature
            if s > best_logit:
                best_logit, best_c = s, c
        prob = sigmoid(best_logit)
        if prob < prob_thr:
            continue
        proposals.append([x0, y0, x1, y1, best_c, prob])

    picked = nms(proposals, nms_thr)
    out = []
    for i in picked:
        o = proposals[i]
        bx0, by0 = o[0] / scale, o[1] / scale
        bx1, by1 = o[2] / scale, o[3] / scale
        bx0 = max(0.0, min(bx0, img_w - 1))
        by0 = max(0.0, min(by0, img_h - 1))
        bx1 = max(0.0, min(bx1, img_w - 1))
        by1 = max(0.0, min(by1, img_h - 1))
        out.append((names[o[4]], round(o[5], 3), (bx0, by0, bx1, by1)))
    return out


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def main():
    E = 8  # 测试用小维度
    A = 5  # 5 个候选框
    names = ["person", "car", "dog"]
    # 提示词嵌入（已 L2 归一化），person/car/dog 各不同
    rng = __import__("random").Random(42)
    text_emb = []
    for _ in names:
        v = _norm([rng.gauss(0, 1) for _ in range(E)])
        text_emb.extend(v)

    # 构造 box 张量 (x1,y1,x2,y2 在 640 空间)
    boxes = [
        [100, 100, 200, 300],   # person-like
        [100, 100, 200, 300],   # 与上一个高度重叠 -> NMS 应只留一个
        [400, 400, 500, 480],   # car-like
        [10, 10, 20, 20],       # 低分(弱响应) -> 过滤
        [300, 300, 360, 360],   # dog-like
    ]
    # 构造 cls 嵌入: 让每个框的视觉特征接近对应类别的文本嵌入
    # 注意: YOLO-World 背景锚点的视觉特征约为零向量 -> 对所有类打分≈0.5(floor)。
    # 因此开放词汇检测阈值通常要略高于 0.5 才能抑制背景。
    cls = []
    for i, b in enumerate(boxes):
        if i == 3:
            f = [0.0] * E  # 背景锚点: 零特征 -> 0.5 floor, 高于阈值被过滤
        else:
            # 视觉特征 = 对应类别文本嵌入 * 增益 + 噪声
            cls_idx = 0 if i < 2 else (1 if i == 2 else 2)
            base = text_emb[cls_idx * E:(cls_idx + 1) * E]
            f = _norm([base[k] * 3.0 + rng.gauss(0, 0.2) for k in range(E)])
        cls.append(f)

    PROB_THR = 0.51  # 略高于 YOLO-World 背景 0.5 floor

    # 测试两种布局
    for baf, caf in [(True, True), (False, False)]:
        box_t = boxes if baf else [[boxes[r][c] for r in range(A)] for c in range(4)]
        cls_t = cls if caf else [[cls[r][c] for r in range(A)] for c in range(E)]
        res = decode(box_t, cls_t, text_emb, names, 640, 1280, 720, prob_thr=PROB_THR)
        print("layout box_anchor_first=%s cls_anchor_first=%s ->" % (baf, caf))
        for r in res:
            print("   ", r)

    # 断言
    res = decode(boxes, cls, text_emb, names, 640, 1280, 720, prob_thr=PROB_THR)
    labels = [r[0] for r in res]
    assert labels.count("person") == 1, "NMS 应合并重复 person 框"
    assert "car" in labels and "dog" in labels, "应检出 car 与 dog"
    assert len(res) == 3, "弱响应框应被过滤, 期望 3 个: %r" % res
    print("\nALL TESTS PASSED ✅  (检出 %d 个目标: %s)" % (len(res), labels))


if __name__ == "__main__":
    main()

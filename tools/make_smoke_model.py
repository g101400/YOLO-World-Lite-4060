#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 YOLO-World Lite demo 的 smoke-test ncnn 模型
================================================

作用：在**还没有真实 YOLO-World 权重**的情况下，让 APK 一装上就能跑通整条链路
（相机采集 → ncnn 推理 → 锚点解码 → 文本嵌入相似度打分 → NMS → 画框），
从而把"代码/构建/渲染/文本塔 HTTP"这些集成问题一次性隔离出来。

⚠️ 这不是真实检测器。它内部只有一个手工搭的极简网络：
   - stem: 160x160 大核卷积，把每 160x160 像素块压成 16 维描述子
           (ch0=全局亮度, ch1=水平梯度, ch2=垂直梯度, ch3..15=随机投影)
   - box_head: 由梯度算出框 (所以框会随画面亮暗梯度轻微移动)
   - cls_head: feat = s*u*(亮度-0.45)，与文本嵌入点积后再 sigmoid
                => 只有"较亮"的像素块才会超过阈值，形成稀疏的演示框
所以它的"检测结果"只是亮度/梯度驱动的玩具输出，仅用于验证链路，不要当精度用。
真实权重请按 README.md 用 convert_yoloworld_ncnn.py 自行导出。

网络 I/O 契约（与 yoloworld.cpp 解码器一致，可用真实模型替换）:
    输入  images   [3, 640, 640]  float, 已 /255
    输出0 output0  ncnn dims (w=A, h=4,   c=1)   -> 每锚点 (x1,y1,x2,y2)
    输出1 output1  ncnn dims (w=A, h=E,   c=1)   -> 每锚点 E 维视觉嵌入
    A=6400 也可以，解码器从张量元素数自动推断 A 与 E。

用法:
    python3 tools/make_smoke_model.py                 # 生成到 app/src/main/assets/
    python3 tools/make_smoke_model.py --verify        # 生成 + 用 ncnn 跑一遍自检
    python3 tools/make_smoke_model.py --verify --preview smoke_preview.png
    python3 tools/make_smoke_model.py --out /tmp      # 只生成到指定目录

自检需要 `pip install ncnn numpy`（Python 端 ncnn 运行时，用于验证 param/bin 能被
ncnn 正确加载与推理）。没有装的话脚本会跳过自检并提示。
"""
import argparse
import hashlib
import math
import os
import random
import struct
import sys

import numpy as np

# ---- 网络结构常量（改这里就能换布局） ----
TARGET_SIZE = 640
KERNEL = 160          # stem 核大小
STRIDE = 160          # stem 步长  -> grid = (640-160)/160+1 = 4  -> A = 16
STEM_C = 16           # stem 输出通道
EMBED_DIM = 512       # 与 CLIP ViT-B/32 / 文本塔服务一致
CLS_SCALE = 3.0       # cls 头缩放 s，控制打分分布（越大越饱和）
BRIGHT_CENTER = 0.45  # 阈值中心：亮度 < 0.45 的块会被压到阈值以下

GRID = (TARGET_SIZE - KERNEL) // STRIDE + 1     # 4
NUM_ANCHORS = GRID * GRID                        # 16
STEM_WEIGHT_NUM = STEM_C * 3 * KERNEL * KERNEL   # 1,228,800

# 与 Android 端一致的阈值
PROB_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45
TEMPERATURE = 0.07


# --------------------------------------------------------------------------- #
# 1. 参数文件 (.param)
# --------------------------------------------------------------------------- #
def build_param_text():
    return "\n".join([
        "7767517",
        "7 8",
        # type            name         in out  blobs...                         params
        "Input            images       0 1 images",
        "Convolution      stem         1 1 images stem  0=%d 1=%d 3=%d 5=1 6=%d"
        % (STEM_C, KERNEL, STRIDE, STEM_WEIGHT_NUM),
        "Split            stem_split   1 2 stem stem_a stem_b",
        "Convolution      box_head     1 1 stem_a box_pre 0=4 1=1 5=1 6=%d" % (4 * STEM_C),
        "Reshape          box_reshape  1 1 box_pre output0 0=%d 1=4 2=1" % NUM_ANCHORS,
        "Convolution      cls_head     1 1 stem_b cls_pre 0=%d 1=1 5=1 6=%d"
        % (EMBED_DIM, EMBED_DIM * STEM_C),
        "Reshape          cls_reshape  1 1 cls_pre output1 0=%d 1=%d 2=1" % (NUM_ANCHORS, EMBED_DIM),
        "",
    ])


# --------------------------------------------------------------------------- #
# 2. 权重 (.bin)  —— 顺序必须与 param 中 layer 出现顺序一致
# --------------------------------------------------------------------------- #
def build_weights(seed=4060, cls_scale=None):
    rng = np.random.RandomState(seed)
    if cls_scale is None:
        cls_scale = CLS_SCALE
    patch = 3 * KERNEL * KERNEL          # 76800
    half = 3 * KERNEL * (KERNEL // 2)    # 38400 = 一半像素数

    # --- stem: (STEM_C, 3, KERNEL, KERNEL) ---
    stem_w = np.zeros((STEM_C, 3, KERNEL, KERNEL), dtype=np.float32)

    # ch0 = 全局平均亮度
    stem_w[0, :, :, :] = 1.0 / patch
    # ch1 = 水平梯度  (左半 +, 右半 -)  -> L̄ - R̄ ∈ [-1,1]
    stem_w[1, :, :, :KERNEL // 2] = 1.0 / half
    stem_w[1, :, :, KERNEL // 2:] = -1.0 / half
    # ch2 = 垂直梯度  (上半 +, 下半 -)  -> T̄ - B̄ ∈ [-1,1]
    stem_w[2, :, :KERNEL // 2, :] = 1.0 / half
    stem_w[2, :, KERNEL // 2:, :] = -1.0 / half
    # ch3..15 = 随机投影，给 cls 头一点额外变化
    stem_w[3:, :, :, :] = rng.normal(0.0, 1.0 / math.sqrt(patch), size=(STEM_C - 3, 3, KERNEL, KERNEL))
    stem_b = np.zeros((STEM_C,), dtype=np.float32)

    # --- box_head: (4, STEM_C, 1, 1) ---
    # x1 = 220 + 60*g_h ; x2 = x1 + 60
    # y1 = 180 + 60*g_v ; y2 = y1 + 45
    box_w = np.zeros((4, STEM_C, 1, 1), dtype=np.float32)
    box_b = np.zeros((4,), dtype=np.float32)
    box_w[0, 1] = 60.0
    box_b[0] = 220.0
    box_w[1, 2] = 60.0
    box_b[1] = 180.0
    box_w[2, 1] = 60.0
    box_b[2] = 280.0
    box_w[3, 2] = 60.0
    box_b[3] = 225.0

    # --- cls_head: (EMBED_DIM, STEM_C, 1, 1) ---
    # feat_j = s*u_j*(brightness - BRIGHT_CENTER)
    u = rng.normal(0.0, 1.0, size=EMBED_DIM).astype(np.float32)
    u /= np.linalg.norm(u)
    cls_w = np.zeros((EMBED_DIM, STEM_C, 1, 1), dtype=np.float32)
    cls_w[:, 0, 0, 0] = cls_scale * u
    cls_b = (-BRIGHT_CENTER * cls_scale * u).astype(np.float32)

    return [(stem_w, stem_b), (box_w, box_b), (cls_w, cls_b)]


def write_model(out_dir, seed=4060, cls_scale=None):
    os.makedirs(out_dir, exist_ok=True)
    param_path = os.path.join(out_dir, "yoloworld.param")
    bin_path = os.path.join(out_dir, "yoloworld.bin")

    with open(param_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_param_text())

    # ncnn .bin 布局（见 ncnn src/modelbin.cpp ModelBinFromDataReader::load）:
    #   权重 blob (type 0): [4 字节 flag][数据]   flag=0 表示 float32
    #   bias   blob (type 1): [数据]              不带 flag
    with open(bin_path, "wb") as f:
        for w, b in build_weights(seed, cls_scale):
            f.write(struct.pack("<I", 0))                            # flag 0 = float32
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())  # bias 无 flag

    return param_path, bin_path


# --------------------------------------------------------------------------- #
# 3. 自检：用 ncnn 加载 → 推理 → 复刻 C++ 解码 → 报告结果
# --------------------------------------------------------------------------- #
def mock_embedding(text, dim=EMBED_DIM):
    """与 yoloworld_text_tower.py 的 --mock 完全一致。"""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return np.array([x / n for x in v], dtype=np.float32)


def make_test_images():
    """返回 [(名称, HxWx3 uint8 RGB)]，覆盖暗/中/亮/亮斑/渐变。"""
    h = w = TARGET_SIZE
    imgs = []
    for name, val in [("dark_0.12", 0.12), ("mid_0.45", 0.45), ("bright_0.85", 0.85)]:
        imgs.append((name, np.full((h, w, 3), int(val * 255), dtype=np.uint8)))

    blob = np.full((h, w, 3), 40, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - 200) ** 2 + (yy - 250) ** 2 < 110 ** 2
    blob[mask] = 235
    imgs.append(("bright_blob", blob))

    grad = np.zeros((h, w, 3), dtype=np.uint8)
    grad[:, :, :] = np.linspace(20, 235, w, dtype=np.uint8)[None, :, None]
    imgs.append(("h_gradient", grad))

    # 类"室内场景"：亮窗 + 桌面 + 暗椅子 + 噪声（非正方形，顺便测 letterbox）
    rng = np.random.RandomState(7)
    scene = np.full((480, 640, 3), 70, dtype=np.uint8)
    scene[60:200, 420:610] = 240
    scene[320:460, 40:360] = 160
    scene[240:420, 380:520] = 35
    scene = np.clip(scene.astype(np.int16) + rng.randint(-12, 12, scene.shape), 0, 255).astype(np.uint8)
    imgs.append(("scene_640x480", scene))
    return imgs


def letterbox_bgr_to_mat(ncnn, rgb):
    """复刻 C++: from_pixels_resize + copy_make_border(114) + /255 归一化。"""
    h, w = rgb.shape[:2]
    scale = float(TARGET_SIZE) / max(w, h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    rw, rh = min(rw, TARGET_SIZE), min(rh, TARGET_SIZE)

    # ncnn 的 from_pixels 用 PIXEL_RGB 直接吃 RGB 字节
    data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    m = ncnn.Mat.from_pixels_resize(data, ncnn.Mat.PixelType.PIXEL_RGB, w, h, rw, rh)

    padded = ncnn.Mat()
    ncnn.copy_make_border(m, padded, 0, TARGET_SIZE - rh, 0, TARGET_SIZE - rw,
                          ncnn.BorderType.BORDER_CONSTANT, 114.0)
    padded.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
    return padded


def decode_like_cpp(box_mat, cls_mat, names, embs, img_w, img_h, scale):
    """逐行复刻 yoloworld.cpp 的 detect()。返回 objects 列表。"""
    total_box = box_mat.w * box_mat.h
    num_anchors = total_box // 4
    total_cls = cls_mat.w * cls_mat.h
    embed_dim = total_cls // num_anchors
    box_anchor_first = (box_mat.h == num_anchors)
    cls_anchor_first = (cls_mat.h == embed_dim)

    def get_box4(a):
        out = np.zeros(4, dtype=np.float32)
        if box_mat.h == 4:
            for k in range(4):
                out[k] = box_mat.row(k)[a]
        else:
            p = box_mat.row(a)
            for k in range(4):
                out[k] = p[k]
        return out

    def get_feat(a):
        out = np.zeros(embed_dim, dtype=np.float32)
        if cls_mat.h == embed_dim:
            for k in range(embed_dim):
                out[k] = cls_mat.row(k)[a]
        else:
            p = cls_mat.row(a)
            for k in range(embed_dim):
                out[k] = p[k]
        return out

    proposals = []
    for a in range(num_anchors):
        b = get_box4(a)
        feat = get_feat(a)
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        logits = embs.dot(feat) / TEMPERATURE
        c = int(np.argmax(logits))
        prob = 1.0 / (1.0 + math.exp(-float(logits[c])))
        if prob < PROB_THRESHOLD:
            continue
        proposals.append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, label=c, prob=prob))

    proposals.sort(key=lambda o: -o["prob"])
    picked = []
    for i, a in enumerate(proposals):
        keep = True
        for j in picked:
            b = proposals[j]
            ix1 = max(a["x"], b["x"])
            iy1 = max(a["y"], b["y"])
            ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
            iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            uni = a["w"] * a["h"] + b["w"] * b["h"] - inter
            if uni > 0 and inter / uni > NMS_THRESHOLD:
                keep = False
                break
        if keep:
            picked.append(i)

    objs = []
    for i in picked:
        o = dict(proposals[i])
        bx0 = min(max(o["x"] / scale, 0.0), img_w - 1)
        by0 = min(max(o["y"] / scale, 0.0), img_h - 1)
        bx1 = min(max((o["x"] + o["w"]) / scale, 0.0), img_w - 1)
        by1 = min(max((o["y"] + o["h"]) / scale, 0.0), img_h - 1)
        o.update(x=bx0, y=by0, w=bx1 - bx0, h=by1 - by0,
                 name=names[o["label"]] if o["label"] < len(names) else "?")
        objs.append(o)
    return objs, num_anchors, embed_dim, box_anchor_first, cls_anchor_first


def verify(param_path, bin_path, preview_path=None):
    try:
        import ncnn
    except ImportError:
        print("[verify] 跳过：未安装 Python 版 ncnn（pip install ncnn numpy）")
        return False

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    if net.load_param(param_path) != 0:
        print("[verify] FAIL: load_param 失败 —— param 语法有问题")
        return False
    if net.load_model(bin_path) != 0:
        print("[verify] FAIL: load_model 失败 —— bin 大小/顺序有问题")
        return False
    print("[verify] ncnn 加载成功: inputs=%s outputs=%s"
          % (list(net.input_names()), list(net.output_names())))

    names = ["person", "car", "dog", "cup", "chair", "phone"]
    embs = np.stack([mock_embedding(n) for n in names])

    ok = True
    for img_name, rgb in make_test_images():
        h, w = rgb.shape[:2]
        scale = float(TARGET_SIZE) / max(w, h)

        ex = net.create_extractor()
        ex.input("images", letterbox_bgr_to_mat(ncnn, rgb))
        r0, box_mat = ex.extract("output0")
        r1, cls_mat = ex.extract("output1")
        if r0 != 0 or r1 != 0:
            print("[verify] FAIL: extract 失败 (%s)" % img_name)
            ok = False
            continue

        objs, A, E, baf, caf = decode_like_cpp(box_mat, cls_mat, names, embs, w, h, scale)
        print("[verify] %-14s output0=(w=%d,h=%d,c=%d) output1=(w=%d,h=%d,c=%d) A=%d E=%d "
              "box_anchor_first=%s cls_anchor_first=%s -> 检出 %d"
              % (img_name, box_mat.w, box_mat.h, box_mat.c, cls_mat.w, cls_mat.h, cls_mat.c,
                 A, E, baf, caf, len(objs)))

        if A != NUM_ANCHORS or E != EMBED_DIM:
            print("        ! 期望 A=%d E=%d，实际不符" % (NUM_ANCHORS, EMBED_DIM))
            ok = False

        if preview_path and img_name == "scene_640x480":
            try:
                import cv2
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                for o in objs:
                    p0 = (int(o["x"]), int(o["y"]))
                    p1 = (int(o["x"] + o["w"]), int(o["y"] + o["h"]))
                    cv2.rectangle(vis, p0, p1, (0, 200, 255), 2)
                    cv2.putText(vis, "%s %.2f" % (o["name"], o["prob"]),
                                (p0[0], max(12, p0[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                cv2.imwrite(preview_path, vis)
                print("        预览图已写出: %s" % preview_path)
            except ImportError:
                print("        (未装 opencv-python，跳过预览图)")

    print("[verify] %s" % ("全部通过 ✅" if ok else "存在问题 ❌"))
    return ok


# --------------------------------------------------------------------------- #
def sweep(tmp_dir, scales, seed=4060):
    """扫一下 cls 头缩放 s，看不同 s 下每个测试图能出多少个框（挑一个观感合适的）。"""
    try:
        import ncnn
    except ImportError:
        print("[sweep] 跳过：未装 Python 版 ncnn")
        return

    names = ["person", "car", "dog", "cup", "chair", "phone"]
    embs = np.stack([mock_embedding(n) for n in names])
    imgs = make_test_images()

    print("[sweep] %-12s | %s" % ("cls_scale s", " ".join("%-14s" % n for n, _ in imgs)))
    for s in scales:
        p, b = write_model(tmp_dir, seed, cls_scale=s)
        net = ncnn.Net()
        net.load_param(p)
        net.load_model(b)
        counts = []
        for _, rgb in imgs:
            h, w = rgb.shape[:2]
            scale = float(TARGET_SIZE) / max(w, h)
            ex = net.create_extractor()
            ex.input("images", letterbox_bgr_to_mat(ncnn, rgb))
            _, box_mat = ex.extract("output0")
            _, cls_mat = ex.extract("output1")
            objs, _, _, _, _ = decode_like_cpp(box_mat, cls_mat, names, embs, w, h, scale)
            counts.append(len(objs))
        print("[sweep] %-12s | %s" % (s, " ".join("%-14d" % c for c in counts)))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(here), "app", "src", "main", "assets")
    ap.add_argument("--out", default=default_out, help="输出目录 (默认 android assets)")
    ap.add_argument("--seed", type=int, default=4060)
    ap.add_argument("--verify", action="store_true", help="生成后用 ncnn 跑自检")
    ap.add_argument("--preview", default=None, help="自检时输出一张标注预览图")
    ap.add_argument("--sweep", action="store_true", help="扫描 cls 头缩放 s 对框数量的影响")
    args = ap.parse_args()

    if args.sweep:
        tmp = os.path.join(here, "_sweep_tmp")
        sweep(tmp, [1.0, 2.0, 3.0, 4.0, 5.0, 8.0], args.seed)
        return

    param_path, bin_path = write_model(args.out, args.seed)
    print("[gen] grid=%dx%d  A=%d  E=%d  stem_weights=%d"
          % (GRID, GRID, NUM_ANCHORS, EMBED_DIM, STEM_WEIGHT_NUM))
    print("[gen] %s (%d bytes)" % (param_path, os.path.getsize(param_path)))
    print("[gen] %s (%d bytes)" % (bin_path, os.path.getsize(bin_path)))

    if args.verify:
        ok = verify(param_path, bin_path, args.preview)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把官方 YOLO-World 权重转成此 Demo 所需的 ncnn 模型
===================================================

本 Demo 的 native 解码器 (app/src/main/jni/yoloworld.cpp) 期望的 ncnn 模型约定：
    输入  : blob 名 "images"      shape [1, 3, H, W]  (H=W=640 常见)
    输出0 : blob 名 "output0"     boxes  形状 [1, 4, A] 或 [1, A, 4]  (A=锚点数, 例如 840)
                                 —— (x1,y1,x2,y2) 绝对坐标(已 letterbox 到输入尺寸)
    输出1 : blob 名 "output1"     cls    形状 [1, E, A] 或 [1, A, E]  (E=文本嵌入维, 例如 512)
                                 —— 视觉对齐嵌入, 与 CLIP 文本嵌入做相似度

解码器会自动识别 [A,4]/[4,A] 与 [A,E]/[E,A] 布局, 无需手工指定。

下面给出两条主流导出路线 + 一键转换脚本。需要一台有 GPU/网络的机器(本沙箱网络受限,
故此处只提供可复用的脚本与命令, 在你自己的机器上运行即可)。

路线 A1 (ultralytics, 推荐, 最简单):
    pip install ultralytics onnx onnxsim
    yolo export model=yoloworld-v2s.pt format=onnx imgsz=640   # 得到 yoloworld-v2s.onnx
    # 再跑本脚本: python convert_yoloworld_ncnn.py yoloworld-v2s.onnx

路线 A2 (mmyolo / 官方 mmdet, 最贴近论文):
    # 用 mmyolo 的 YOLOWorld 导出脚本得到 onnx, 再跑本脚本转 ncnn
    python demo/export_onnx.py \
        configs/yolo_world/yoloworld_v2_s_vlpan_bn_2e-3_100e_coco.py \
        yoloworld_v2_s_pretrained.pth --out yoloworld.onnx

ncnn 转换依赖 onnx2ncnn (来自你本机编译的 ncnn, 或 pip 的 ncnn 包)。
"""
import os
import sys


def step_simplify(onnx_in, onnx_out):
    import onnx
    import onnxsim
    model = onnx.load(onnx_in)
    model_simplified, _ = onnxsim.simplify(model)
    onnx.save(model_simplified, onnx_out)
    print("[convert] onnx simplified ->", onnx_out)


def step_onnx2ncnn(onnx_path, out_dir, onnx2ncnn_bin):
    # onnx2ncnn <in.onnx> <out.param> <out.bin>
    import subprocess
    param = os.path.join(out_dir, "yoloworld.param")
    binf = os.path.join(out_dir, "yoloworld.bin")
    subprocess.check_call([onnx2ncnn_bin, onnx_path, param, binf])
    print("[convert] ncnn ->", param, binf)
    return param, binf


def sanity_check(param_path):
    """打印 ncnn param 的 blob 形状, 确认与本 Demo 约定一致。"""
    import re
    shapes = {}
    with open(param_path, "r") as f:
        for line in f:
            # 形如: Convolution ... 0 1 in0 0 0 0 ... 或 更常见的 blob 行
            m = re.search(r"(\w[\w-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line.strip())
            if m:
                shapes[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    print("[convert] 检测到 blob 形状(仅供参考):")
    for k, v in shapes.items():
        print("   ", k, v)
    print("[convert] 请确认存在 output0(boxes) 与 output1(cls), 输入名为 images。")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python convert_yoloworld_ncnn.py <model.onnx> [--onnx2ncnn /path/to/onnx2ncnn]")
        return
    onnx_in = sys.argv[1]
    onnx2ncnn_bin = None
    if "--onnx2ncnn" in sys.argv:
        onnx2ncnn_bin = sys.argv[sys.argv.index("--onnx2ncnn") + 1]

    out_dir = os.path.dirname(os.path.abspath(onnx_in))
    onnx_sim = os.path.join(out_dir, "yoloworld.sim.onnx")
    step_simplify(onnx_in, onnx_sim)

    if onnx2ncnn_bin:
        param, binf = step_onnx2ncnn(onnx_sim, out_dir, onnx2ncnn_bin)
        sanity_check(param)
        print("\n[convert] 完成! 把 yoloworld.param / yoloworld.bin 拷到:")
        print("   app/src/main/assets/  然后重新构建 APK。")
    else:
        print("[convert] 未提供 --onnx2ncnn, 仅完成 onnx 简化。")
        print("   请用你本机编译的 ncnn 的 onnx2ncnn 转换:")
        print("   onnx2ncnn %s yoloworld.param yoloworld.bin" % onnx_sim)


if __name__ == "__main__":
    main()

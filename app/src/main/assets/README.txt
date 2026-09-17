把 YOLO-World 的 ncnn 模型放到本目录，文件名必须为：

    yoloworld.param
    yoloworld.bin

获取方式见仓库根目录的 convert_yoloworld_ncnn.py / README.md
（用 ultralytics 或 mmyolo 导出 YOLO-World onnx，再用 onnx2ncnn 转成本地 ncnn）。

未放置模型时，App 会提示“未加载模型”，不会崩溃。

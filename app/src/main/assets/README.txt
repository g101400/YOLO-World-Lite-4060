本目录当前放的是 **smoke-test 模型**（由 tools/make_smoke_model.py 生成，约 5MB），
装上 APK 即可直接跑通整条链路：相机 → ncnn 推理 → 解码 → 文本嵌入相似度 → NMS → 画框。

⚠️ 它不是真实检测器！内部只有 3 层卷积，"检测结果"是亮度/梯度驱动的玩具输出，
   仅用于验证 App 端到端链路与文本塔服务联通。

换成真实 YOLO-World 权重：
    1) 在你自己的机器上按仓库根 README.md / convert_yoloworld_ncnn.py 导出 onnx → ncnn
    2) 用生成的 yoloworld.param / yoloworld.bin 覆盖本目录同名文件
    3) 重新 ./gradlew.bat assembleRelease

文件名必须为：
    yoloworld.param
    yoloworld.bin

未放置模型时，App 会提示"no model"，不会崩溃。

本目录当前放的是 **smoke-test 模型**（由 tools/make_smoke_model.py 生成，约 5MB），
装上 APK 即可直接跑通整条链路：相机 → ncnn 推理 → 解码 → 文本嵌入相似度 → NMS → 画框 → 中文标签。

⚠️ 它不是真实检测器！内部只有 3 层卷积，"检测结果"是亮度/梯度驱动的玩具输出，
   仅用于验证 App 端到端链路。

文件名必须为：
    yoloworld.param
    yoloworld.bin

换成真实 YOLO-World 权重：
    1) 在你自己的机器上按仓库根 README.md / convert_yoloworld_ncnn.py 导出 onnx → ncnn
    2) 用生成的 yoloworld.param / yoloworld.bin 覆盖本目录同名文件
    3) 重新 ./gradlew.bat assembleRelease

未放置模型时，App 画面会显示中文提示"未加载模型…"，不会崩溃。

--------------------------------------------------------------------------
可选：prompt_emb.bin（端侧嵌入词表）
--------------------------------------------------------------------------
App 自带一个端侧离线编码器，任何提示词都能用、不需要服务，但它不是 CLIP 对齐的。
若要用真实 CLIP 语义又不想让手机连 PC，就把固定词表的嵌入固化进 APK：

    python yoloworld_text_tower.py --clip                     # 另开终端
    python tools/build_prompt_table.py --prompts "人,汽车,自行车,狗" \
           --server http://127.0.0.1:8000
    # 产物即本目录的 prompt_emb.bin，重新编译后 App 自动加载

放了这个文件后，App 状态栏会显示"端侧词表 N 条"。格式见 tools/build_prompt_table.py 顶部注释。

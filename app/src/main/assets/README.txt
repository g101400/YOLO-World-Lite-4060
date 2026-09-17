assets 目录内容
==============

yoloworld.param / yoloworld.bin
  正式 YOLO-World (yolov8s-worldv2) ncnn 权重，已随包发布。
  - 输入: images [1,3,640,640](RGB/255, letterbox 114 填充) + txt [1,10,512]
  - 输出: output0 boxes [1,4,8400] (xyxy, letterbox 640 坐标系)
          output1 scores [1,10,8400] (sigmoid 后的类别分数)
  - 来源: 官方 ultralytics 权重经 tools/export_real_weights.py 导出 (pnnx)

架构注意: 骨干含 C2fAttn 文本引导注意力，文本嵌入是网络输入而非仅参与解码，
因此提示词嵌入必须运行时喂入 (App 内置离线编码器或词表/远端服务)。

prompt_emb.bin (可选)
  预计算的 CLIP 文本嵌入词表，用 tools/build_prompt_table.py 生成。
  真实语义请用 CLIP (yoloworld_text_tower.py --clip) 生成，与模型训练一致。

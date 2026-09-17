# YOLO-World-Lite · 开放词汇检测（路线 A）

把固定 80 类的 YOLO-Lite 检测器，升级为 **用任意文本提示词即可检测** 的开放词汇检测器。
基于 YOLO-World（YOLOv8 同族架构），复用 v8/v11/v26 的 ncnn 解码器，检测完全在手机端（ncnn），
类别由「视觉嵌入 · CLIP 文本嵌入」相似度决定。

> 本仓库对应“四系列扩展 80 分类 → 开放词汇”的 **路线 A**。详见 `docs/yoloworld_route_a.html`。

---

## 目录结构

```
android_demo/ncnn-android-yoloworld/
├── app/src/main/
│   ├── java/ncnn/yoloworld/demo/
│   │   ├── MainActivity.java        # 提示词 UI + 相机/权限
│   │   ├── NcnnYoloworld.java       # JNI 桥
│   │   ├── YoloOverlayView.java     # 预览层: 画框 + 中文标签 + 中文状态提示
│   │   └── TextEmbedder.java        # 端侧文本编码(词表 / 内置离线编码器)
│   ├── jni/
│   │   ├── yoloworld.h / .cpp       # 解码器: 锚点无关 + 文本嵌入相似度
│   │   ├── yoloworldncnn.cpp        # JNI + 渲染循环(只算不画，结果回调给 Java)
│   │   ├── ywmimage.h               # 零依赖图像容器 (替代 OpenCV)
│   │   ├── ndkcamera.h / .cpp       # Camera2 NDK 采集与渲染
│   │   └── CMakeLists.txt           # 仅链接 ncnn (E:/AndroidSDK)，不依赖 OpenCV
│   ├── assets/                      # yoloworld.param / .bin，可选 prompt_emb.bin
│   └── res/                         # 布局/字符串(全中文)
├── yoloworld_text_tower.py          # CLIP 文本塔服务 (可选，真实/ mock)
├── convert_yoloworld_ncnn.py        # 官方权重 → 本Demo约定ncnn
├── tools/test_decode.py             # 解码算法离线自测
├── tools/make_smoke_model.py        # 生成/自检 smoke-test ncnn 模型(可脱机跑通链路)
├── tools/export_real_weights.py     # 官方 yolov8s-worldv2.pt → ncnn (pnnx, 文本作为第二输入)
├── tools/finish_export.py           # pnnx 产物 blob 重命名 → assets + ncnn 运行时验证
├── tools/build_prompt_table.py      # 生成端侧嵌入词表 assets/prompt_emb.bin
├── docs/ncnn_model_format.md        # ncnn param/bin 格式速查(手写模型/排错)
├── docs/yoloworld_route_a.html      # 路线A 知识文档
└── README.md
```

## 工作原理

```
摄像头帧 → [ncnn 检测骨干] → 每锚点: 框回归 + 视觉嵌入
文本提示词 → [CLIP 文本塔] → 文本嵌入
得分 = sigmoid( (视觉嵌入 · 文本嵌入) / τ ),  τ≈0.07
```

检测骨干与框回归 100% 在端侧。**文本嵌入也在端侧完成**，三级来源（见 `TextEmbedder.java`）：

| 优先级 | 来源 | 说明 |
|---|---|---|
| 1 | `assets/prompt_emb.bin` | 预计算的 CLIP 词表，任意维度、无需联网。用 `tools/build_prompt_table.py` 生成 |
| 2 | 内置离线编码器 | 字符/二元组哈希投影，任意提示词可用、零依赖；**不是 CLIP 对齐**，仅配合验证模型有意义 |
| 3 | 远端文本塔服务 | 勾选「使用远端文本编码服务」后启用，适合跑真实 CLIP 文本塔；失败会自动退回端侧 |

框与标签由 Android 侧 `YoloOverlayView` 用系统字体绘制，所以**中文提示词、中文状态提示都能正常显示**。

## 快速开始

### 0. 直接装 APK 跑通链路（零依赖）
仓库 `app/src/main/assets/` 里已经放了一个 **smoke-test 模型**（`tools/make_smoke_model.py` 生成，约 5MB），
所以 `release/YOLO-World-Lite-v1.0-release.apk` **装上就能跑**，不需要先准备权重：

- 能看到相机画面、画框、类别名与置信度；
- 在 App 内填提示词（中英文都行）、点「应用提示词」，即可看到标签随提示词变化；
- 默认走端侧编码，**不需要开任何服务、不需要联网**。

⚠️ smoke 模型内部只有 3 层卷积，"检测框"是**亮度/梯度驱动的玩具输出**（框会随画面明暗梯度移动、
类别随提示词变化），只用于验证链路，**不代表检测精度**。真实效果必须换正式权重。

重新生成 smoke 模型（可自检）：
```bash
pip install ncnn numpy          # 自检用
python tools/make_smoke_model.py --verify --preview smoke_preview.png
```
自检会打印每个输出的 ncnn 尺寸、按 C++ 解码逻辑得到的框数量，并可选导出标注预览图。

### 1. 准备真实模型（本沙箱网络受限，需在你自己机器上完成）
```bash
# 用 ultralytics 导出 onnx
pip install ultralytics onnx onnxsim
yolo export model=yoloworld-v2s.pt format=onnx imgsz=640
# 转 ncnn（需要你本机编译的 ncnn 的 onnx2ncnn）
python convert_yoloworld_ncnn.py yoloworld-v2s.onnx --onnx2ncnn /path/to/onnx2ncnn
cp yoloworld.param yoloworld.bin app/src/main/assets/
```
ncnn 约定：输入 `images`[1,3,640,640]；输出 `output0`=boxes、`output1`=cls（解码器自动识别布局）。

### 2.（可选）端侧真实语义：固化成词表，或起远端服务

**方式一：把嵌入打进 APK（推荐，手机上完全离线）**
```bash
# 用真实 CLIP 文本塔算好固定词表的嵌入
python yoloworld_text_tower.py --clip                     # 另开终端
python tools/build_prompt_table.py --prompts "人,汽车,自行车,狗" --server http://127.0.0.1:8000
# 产物 app/src/main/assets/prompt_emb.bin，重新编译 APK 即生效
```
不填 `--server` 而用 `--mock` 也能生成（确定性占位向量，只用于验证链路）。

**方式二：远端服务（词表可变时用）**
```bash
python yoloworld_text_tower.py --mock    # 演示(无依赖)
python yoloworld_text_tower.py --clip    # 真实 CLIP 文本塔
```
在 App 里填 `http://<PC局域网IP>:8000` 并勾选「使用远端文本编码服务」。

### 3. 构建并签名 APK（需 E:/AndroidSDK 工具链）
```bash
export JAVA_HOME=E:/jdk11
cd android_demo/ncnn-android-yoloworld
./gradlew.bat assembleRelease
```
`app/build.gradle` 会读取仓库根的 `keystore.properties` 自动完成 v1+v2 签名，产物：
`app/build/outputs/apk/release/com.ncnn.yoloworld.demo-release.apk`

> 注：本机预编译的 opencv-mobile 2.4.13.7 是用更高版本 NDK 构建的（会引用 NDK 21.3 里不存在的
> `__libcpp_verbose_abort` / `__kmpc_dispatch_deinit`），因此本 Demo **完全不链接 OpenCV**，
> 图像缓冲与画框/文字改为 `ywmimage.h` 里的零依赖实现（同时 APK 也小了几 MB）。

### 4. 手机运行
- 安装 APK 后首次启动会申请相机权限，**必须允许**（否则只有黑屏 + 中文提示）；
- 在「类别」框输入提示词（中英文逗号、顿号、分号、换行都可分隔；如 `人, 汽车, 自行车, 狗`），点「应用提示词」；
- 不需要联网、不需要 PC 服务；只有当你要用真实 CLIP 文本塔时才勾选远端服务。

> 版本要求：`minSdk 24 / targetSdk 29`。旧版曾用 `targetSdk 24`，Android 12+ 会提示
> 「此应用专为旧版 Android 打造，可能无法正常运行」，已提升到 29 消除该提示。

## 验证解码算法（无需模型）
```bash
python tools/test_decode.py   # 验证 布局自动识别 / 相似度打分 / NMS / 去letterbox
```

## 调参（app/src/main/jni/yoloworld.h 顶部常量）
| 参数 | 含义 |
|------|------|
| `YW_TEMPERATURE` | 温度 τ，默认 0.07 |
| `YW_PROB_THRESHOLD` | 置信度阈值，开放词汇建议 ≥0.3 |
| `YW_BOX_FORMAT` | 0=(x1,y1,x2,y2) / 1=(cx,cy,w,h) |
| `YW_BLOB_*` | 输入/输出 blob 名，与你的导出一致 |

## 已知约束
- smoke-test 模型只能验证链路，不能给出有意义的检测；真实 YOLO-World ncnn 权重（~100MB+）需按上面步骤自行导出。
- **内置离线编码器不是 CLIP 对齐的**：它保证「不报错、可跑通、任意提示词可用」，但不产生真实语义。
  要真实语义请用 `prompt_emb.bin` 词表（方式一）或远端 CLIP 服务（方式三）。
  真实权重 + 哈希嵌入的组合会得到无意义的结果，务必用词表/服务。
- 远端服务是局域网明文 HTTP，`AndroidManifest.xml` 里已开 `usesCleartextTraffic="true"`（仅此用途）。
- 解码器对分类得分统一做了 `sigmoid`（`YW_CONF_SCALE` 可调）。如果你的导出已经在模型里做过 sigmoid，
  请把 `yoloworld.h` 里的 `YW_CONF_SCALE` 调小或加开关，否则阈值语义会不一致。
- 提示词嵌入维度必须等于模型输出维度：不一致时画面会提示「嵌入维度不匹配」，App 会自动按模型维度重算一次
  （仅端侧编码可用；词表/远端服务会提示手改）。

## v1.1.0 变更
1. `targetSdk 24 → 29`，消除「专为旧版 Android 打造」提示；补 `usesCleartextTraffic`。
2. 绘制从原生搬到 Android `YoloOverlayView`：支持中文标签与中文状态提示，去掉原生 5x7 点阵字。
3. 文本编码下沉到端侧（词表 / 内置离线编码器），不再强制依赖 PC 服务；远端服务变为可选且失败自动降级。
4. 修复首次启动在授予相机权限前就调用 `openCamera()` 的问题（改为授权回调里再打开）。
5. 检测改为在**旋转后的正向缓冲**上执行，框坐标直接是屏幕坐标，Java 覆盖层无需再做旋转换算。

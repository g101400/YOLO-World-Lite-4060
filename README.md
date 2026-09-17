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
│   ├── java/ncnn/yoloworld/demo/   # MainActivity(提示词UI) + NcnnYoloworld(JNI桥)
│   ├── jni/
│   │   ├── yoloworld.h / .cpp       # 解码器: 锚点无关 + 文本嵌入相似度
│   │   ├── yoloworldncnn.cpp        # JNI + 渲染循环
│   │   ├── ywmimage.h               # 零依赖图像容器 + 画框/5x7点阵字 (替代 OpenCV)
│   │   ├── ndkcamera.h / .cpp       # Camera2 NDK 采集与渲染
│   │   └── CMakeLists.txt           # 仅链接 ncnn (E:/AndroidSDK)，不依赖 OpenCV
│   ├── assets/                      # 放入 yoloworld.param / yoloworld.bin
│   └── res/                         # 布局/字符串(提示词输入框、文本塔地址)
├── yoloworld_text_tower.py          # CLIP 文本塔服务 (真实/ mock)
├── convert_yoloworld_ncnn.py        # 官方权重 → 本Demo约定ncnn
├── tools/test_decode.py             # 解码算法离线自测
├── docs/yoloworld_route_a.html      # 路线A 知识文档
└── README.md
```

## 工作原理

```
摄像头帧 → [ncnn 检测骨干] → 每锚点: 框回归 + 视觉嵌入
文本提示词 → [CLIP 文本塔] → 文本嵌入
得分 = sigmoid( (视觉嵌入 · 文本嵌入) / τ ),  τ≈0.07
```

检测骨干与框回归 100% 在端侧；文本塔只在提示词变化时算一次，可用本机 Python 服务或云端提供。

## 快速开始

### 0. 直接装 APK 跑通链路（零依赖）
仓库 `app/src/main/assets/` 里已经放了一个 **smoke-test 模型**（`tools/make_smoke_model.py` 生成，约 5MB），
所以 `release/YOLO-World-Lite-v1.0-release.apk` **装上就能跑**，不需要先准备权重：

- 能看到相机画面、画框、类别名与置信度；
- 在 App 内填提示词、点「应用提示词」，能验证 文本塔 HTTP → 嵌入 → 打分 → 标签变化 的整条通路。

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

### 2. 启动文本塔服务（PC 端）
```bash
python yoloworld_text_tower.py --mock   # 演示(无依赖)
python yoloworld_text_tower.py --clip    # 真实 CLIP 文本塔
```

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
- 把 APK 安装到手机；
- 在 App 内「文本编码器地址」填 `http://<PC局域网IP>:8000`（模拟器填 `http://10.0.2.2:8000`）；
- 在「类别」框输入英文逗号分隔的提示词（如 `person, car, dog`），点「应用提示词」。

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
- 文本塔默认用确定性 mock 嵌入以打通链路；真实检测请用 `--clip`，并尽量使用与 YOLO-World 训练一致的 CLIP 文本塔以保证对齐。
- 端侧 overlay 用的是内置 5x7 点阵字体，只覆盖 ASCII；**中文提示词在画面上会显示为方块**（App 界面中文由 Android 渲染，不受影响）。建议提示词用英文，如 `person, car, dog`。
- 解码器对分类得分统一做了 `sigmoid`（`YW_CONF_SCALE` 可调）。如果你的导出已经在模型里做过 sigmoid，
  请把 `yoloworld.h` 里的 `YW_CONF_SCALE` 调小或加开关，否则阈值语义会不一致。

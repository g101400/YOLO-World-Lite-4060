# 手工生成 / 核对 ncnn 模型文件（param + bin）速查

本文记录的是**实测 + ncnn 源码比对**得到的事实，用于「没有 onnx2ncnn 工具时手写一个 ncnn 模型」
或排查「模型能 load 但推理结果是垃圾」这类问题。踩过的坑都在最后一节。

## 1. `.param` 文本格式

```
7767517
<layer_count> <blob_count>
<type> <name> <in_count> <out_count> <in_blobs...> <out_blobs...> <k=v ...>
...
```

- 第一行固定魔数 `7767517`。
- `layer_count` = 后面实际的层行数。
- `blob_count` = **所有被产出的 blob 名去重后的个数**（不含只被消费的）。
- `Input` 层写 `Input images 0 1 images` 即可，**不需要**写 `0=640 1=640 2=3`，尺寸由运行时输入的 Mat 决定。
- 想要一个 layer 的 blob 被两处消费，要显式插 `Split` 层（如 ncnnoptimize 会插入的那样）。

### Convolution 参数 id（实测确认）

| id | 含义 |
|----|------|
| 0 | num_output |
| 1 | kernel_w（kernel_h 默认同值） |
| 2 | dilation_w |
| 3 | stride_w |
| 4 | pad_left（pad_top 默认同值，不写即 0） |
| 5 | **bias_term** |
| 6 | weight_data_size = num_output × in_channels × kh × kw |
| 9 | activation_type（`1` = ReLU；不写即无激活） |

> 注意：任何**不认识的 id 会被 ncnn 静默忽略**，所以「少写」比「写错」安全。
> 只需要 `0 1 3 5 6` 就能定义一个可用的卷积。

### Reshape 参数 id

`0=w 1=h 2=c`（ncnn 的 Mat 维度是 (w, h, c)）。ncnn 直接按新的 (w,h,c) 重新解释同一段内存，
**元素顺序 = c 最外层、h 次之、w 最内层**。这一点对「把卷积输出变成 [A,4] 这种布局」很关键。

例：卷积输出 (w=4, h=4, c=4) → `Reshape 0=16 1=4 2=1` 得到 (w=16, h=4, c=1)，
此时第 k 行 = 第 k 个通道的 4×4 平面展平。

## 2. `.bin` 二进制格式（关键！）

见 `ncnn/src/modelbin.cpp` → `ModelBinFromDataReader::load(w, type)`：

- **`type == 0`（权重 blob）**：先是 **4 字节 flag**，然后才是数据。
  - `flag == 0x01306B47` → float16 数据
  - `flag == 0x01348B83` → bfloat16 数据
  - `flag == 0x000D0021` → int8 量化
  - `flag == 0x0002C056` → float32 + 额外 scale
  - **`flag == 0` → 裸 float32**（自己造模型就用这个）
  - 其他非 0 值 → 走「量化」分支，会去读 256 个 quantization float，几乎必然读失败或错位
- **`type == 1`（bias blob）**：**没有 flag**，直接 `num_output * 4` 字节裸 float32。
- 卷积权重元素顺序：`[oc][ic][ky][kx]`。
- 每个带权重的层按它在 `.param` 里出现的顺序消费 bin（Split/Reshape 这类无权重层不消费）。

## 3. 「模型能加载但输出是垃圾」的经典症状

| 症状 | 原因 |
|------|------|
| 所有输出通道结果几乎一样 | 权重 blob 少写了 4 字节 flag=0，ncnn 把权重数据开头 4 字节当 flag 用，进了量化分支 → 整体错位 |
| load_model 返回非 0，日志 `ModelBin read quantization_value failed` | 同上；`flag != 0` 触发了量化读取 |
| bias 完全没生效（输出≈0） | bias blob 多写了 4 字节 flag；bias 是裸数据，多写的 4 字节会被当成 bias 本身 |
| 输出通道全是同一个值 | 权重 blob 大小写错（`6=weight_data_size` 与 `num_output × wei × kh × kw` 不一致） |

## 4. 没有真机也能验证：用 Python 版 ncnn

```bash
pip install ncnn numpy
```

```python
import ncnn
net = ncnn.Net()
net.load_param("yoloworld.param")     # != 0 说明 param 语法有问题
net.load_model("yoloworld.bin")       # != 0 说明 bin 大小/顺序有问题
net.input_names(), net.output_names()

ex = net.create_extractor()
ex.input("images", mat)               # mat = ncnn.Mat.from_pixels_resize(...)
ret, out = ex.extract("output0")
print(out.w, out.h, out.c)            # 直接确认张量布局
```

配合「设计值反推」特别有效：把某个卷积的权重设成恒等/常量，喂一张纯色图，
就能算出输出的**解析期望值**，和 ncnn 实际输出逐位对比，一次定位是
「权重错位 / bias 丢失 / 维度错」中的哪一种。

## 5. 本项目里的实操脚本

- `tools/make_smoke_model.py` —— 手工生成一个 3 层 ncnn 模型（smoke-test），
  自带 `--verify`（用 Python ncnn 跑一遍并复刻 C++ 解码逻辑）和 `--sweep`
  （扫描超参看框数量变化），可选 `--preview` 输出标注图。
- `tools/test_decode.py` —— 纯离线验证解码数学（布局自动识别 / 相似度 / NMS / 去 letterbox）。

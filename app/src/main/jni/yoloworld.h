// YOLO-World (open-vocabulary) detector — YOLOv8-family anchor-free head
// with a text-aligned embedding branch. Classification is done by comparing
// each anchor's visual embedding with user prompt text embeddings (CLIP).
//
// The decoder auto-detects the output tensor layout, so it works with the
// common YOLO-World ncnn exports regardless of (840,4)/(4,840) transpose.

#ifndef YOLOWORLD_H
#define YOLOWORLD_H

#include <string>
#include <vector>

#include "ywmimage.h"

#include <net.h>

struct Object
{
    YWRect rect;
    int label;
    float prob;
};

// ---- Model-specific knobs (edit to match your ncnn export) ----
// These mirror the values documented in convert_yoloworld_ncnn.py / README.md.
static const int   YW_TARGET_SIZE = 640;     // network input (letterbox target)
// 官方权重架构: 骨干 C2fAttn 以文本为 guide → 文本嵌入必须作为网络输入
// 模型输入: images[1,3,640,640] + txt[1,10,512]; 输出: output0=boxes[1,4,8400], output1=scores[1,10,8400](已 sigmoid)
// 见 tools/export_real_weights.py
static const int   YW_MAX_CLASSES = 10;      // txt 输入固定 10 行, 不足由 setPrompt 循环填充
static const int   YW_EMBED_DIM   = 512;     // CLIP ViT-B/32 文本嵌入维度
static const int   YW_BOX_FORMAT  = 0;       // 0 = (x1,y1,x2,y2) ; 1 = (cx,cy,w,h)
static const char* YW_BLOB_INPUT  = "images";   // network input blob name
static const char* YW_BLOB_TXT    = "txt";      // text embeddings input blob name
static const char* YW_BLOB_BOX    = "output0";  // decoded boxes  blob name
static const char* YW_BLOB_SCORES = "output1";  // per-class scores blob name (sigmoid 已在图内)
static const float YW_PROB_THRESHOLD = 0.30f;
static const float YW_NMS_THRESHOLD   = 0.45f;
static const float YW_CONF_SCALE      = 1.0f;   // multiply final prob if your export already folds a scale

class Yoloworld
{
public:
    Yoloworld();

    // Load model from Android assets (yoloworld.param / yoloworld.bin).
    int load(AAssetManager* mgr, bool use_gpu = false);

    // Provide prompt class names + their L2-normalized CLIP embeddings (flattened:
    // num_classes * embed_dim, row-major). Must be called before detect().
    int setPrompt(const std::vector<std::string>& names, const std::vector<float>& embeddings);

    int detect(const YWMat& rgb, std::vector<Object>& objects,
               float prob_threshold = YW_PROB_THRESHOLD, float nms_threshold = YW_NMS_THRESHOLD);

    bool loaded() const { return ok; }
    bool hasPrompt() const { return has_prompt; }

    // Embedding dim the model actually outputs (discovered on the first detect).
    int modelEmbedDim() const { return model_embed_dim; }
    // True when the supplied prompt embeddings do not match the model's dim.
    bool dimMismatch() const { return dim_mismatch; }

private:
    ncnn::Net net;

    int target_size;
    bool ok;
    bool has_prompt;

    int embed_dim;        // dim of the embeddings supplied via setPrompt()
    int model_embed_dim;  // dim the model outputs (0 until the first detect)
    int num_anchors;      // A
    int num_classes;      // number of prompt classes
    bool dim_mismatch;

    // prompt state
    std::vector<std::string> class_names;
    std::vector<float> text_emb;  // num_classes * embed_dim

    // auto-detected output layouts
    bool box_anchor_first; // box tensor is [A,4] (true) or [4,A] (false)
    bool cls_anchor_first; // cls tensor is [A,E] (true) or [E,A] (false)

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLOWORLD_H

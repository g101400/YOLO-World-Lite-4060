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
static const float YW_TEMPERATURE = 0.07f;   // YOLO-World tau; divide dot-product by this
static const int   YW_BOX_FORMAT  = 0;       // 0 = (x1,y1,x2,y2) ; 1 = (cx,cy,w,h)
static const char* YW_BLOB_INPUT  = "images";   // network input blob name
static const char* YW_BLOB_BOX    = "output0";  // decoded boxes  blob name
static const char* YW_BLOB_CLS    = "output1";  // text-aligned embeddings blob name
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

    int draw(YWMat& rgb, const std::vector<Object>& objects);

    bool loaded() const { return ok; }
    bool hasPrompt() const { return has_prompt; }

private:
    ncnn::Net net;

    int target_size;
    bool ok;
    bool has_prompt;

    int embed_dim;    // E
    int num_anchors;  // A
    int num_classes;  // number of prompt classes

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

// YOLO-World (open-vocabulary) detector implementation.
// Detection backbone + head are on-device (ncnn). The text embeddings come from
// a CLIP text tower (served by yoloworld_text_tower.py, or precomputed) and are
// fed in via setPrompt(). Classification = temperature-scaled dot product between
// each anchor's visual embedding and the prompt text embeddings.

#include "yoloworld.h"

#include <android/asset_manager.h>

#include <benchmark.h>

#include <omp.h>
#include <cpu.h>

#include <cmath>
#include <cstdio>
#include <algorithm>

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b) {
    YWRect inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objs, int left, int right) {
    int i = left, j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j) {
        while (objs[i].prob > p) i++;
        while (objs[j].prob < p) j--;
        if (i <= j) { std::swap(objs[i], objs[j]); i++; j--; }
    }
    if (left < j) qsort_descent_inplace(objs, left, j);
    if (i < right) qsort_descent_inplace(objs, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objs) {
    if (!objs.empty()) qsort_descent_inplace(objs, 0, (int)objs.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objs, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = (int)objs.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objs[i].rect.area();
    for (int i = 0; i < n; i++) {
        const Object& a = objs[i];
        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++) {
            const Object& b = objs[picked[j]];
            float inter = intersection_area(a, b);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (inter / uni > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

// Read the 4 box coordinates for anchor `a` from a [A,4] or [4,A] ncnn Mat.
// ncnn::Mat only exposes row() (no at()), so index via row pointers.
static inline void get_box4(const ncnn::Mat& box, int a, float* out4) {
    if (box.h == 4) {
        // anchors along width: (coord, anchor) -> element (coord=k, anchor=a)
        for (int k = 0; k < 4; k++) out4[k] = box.row(k)[a];
    } else {
        // anchors along height: (anchor, coord) -> row(a) length 4
        const float* p = box.row(a);
        for (int k = 0; k < 4; k++) out4[k] = p[k];
    }
}

// Read the E-dim visual embedding for anchor `a` from a [A,E] or [E,A] ncnn Mat.

static inline float dot(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

Yoloworld::Yoloworld() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    target_size = YW_TARGET_SIZE;
    ok = false;
    has_prompt = false;
    embed_dim = 0;
    num_anchors = 0;
    num_classes = 0;
    box_anchor_first = true;
    cls_anchor_first = true;
}

int Yoloworld::load(AAssetManager* mgr, bool use_gpu) {
    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = use_gpu;
#endif
    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    // Confirm model assets exist before loading (avoid hard crash on missing file).
    AAsset* pa = AAssetManager_open(mgr, "yoloworld.param", AASSET_MODE_STREAMING);
    AAsset* ba = AAssetManager_open(mgr, "yoloworld.bin", AASSET_MODE_STREAMING);
    if (!pa || !ba) {
        if (pa) AAsset_close(pa);
        if (ba) AAsset_close(ba);
        ok = false;
        return -1;
    }
    AAsset_close(pa);
    AAsset_close(ba);

    if (net.load_param(mgr, "yoloworld.param") != 0 ||
        net.load_model(mgr, "yoloworld.bin") != 0) {
        ok = false;
        return -1;
    }
    ok = true;
    return 0;
}

int Yoloworld::setPrompt(const std::vector<std::string>& names, const std::vector<float>& embeddings) {
    if (names.empty() || embeddings.empty()) {
        has_prompt = false;
        return -1;
    }
    if ((int)names.size() > YW_MAX_CLASSES) {
        has_prompt = false;
        return -2;  // 超过模型支持的类别数(txt 输入固定 YW_MAX_CLASSES 行)
    }
    int E = (int)(embeddings.size() / names.size());
    if (E <= 0 || (int)embeddings.size() != (int)names.size() * E) {
        has_prompt = false;
        return -1;
    }
    if (E != YW_EMBED_DIM) {
        has_prompt = false;
        dim_mismatch = true;
        return -3;  // 与模型 txt 输入维度不符
    }
    num_classes = (int)names.size();
    embed_dim = E;
    model_embed_dim = YW_EMBED_DIM;
    class_names = names;
    // L2 归一化每条真实提示词, 再循环填充到 K 行(填充行的分数不会被读取)
    text_emb.assign(YW_MAX_CLASSES * E, 0.f);
    for (int k = 0; k < YW_MAX_CLASSES; k++) {
        int c = k % num_classes;
        const float* src = &embeddings[c * E];
        double n2 = 0.0;
        for (int i = 0; i < E; i++) n2 += (double)src[i] * src[i];
        float n = (float)sqrt(n2);
        float* dst = &text_emb[k * E];
        if (n > 1e-12f)
            for (int i = 0; i < E; i++) dst[i] = src[i] / n;
    }
    has_prompt = true;
    dim_mismatch = false;
    return 0;
}

int Yoloworld::detect(const YWMat& rgb, std::vector<Object>& objects,
                      float prob_threshold, float nms_threshold) {
    objects.clear();
    if (!ok || !has_prompt) return 0;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox resize to target_size, pad to square
    int w = img_w, h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(img_h * scale);
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(img_w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input(YW_BLOB_INPUT, in_pad);

    // 文本嵌入输入 [K,E]: 行 k = 第 k 个(填充后)提示词
    ncnn::Mat txt_mat(YW_EMBED_DIM, YW_MAX_CLASSES);
    for (int k = 0; k < YW_MAX_CLASSES; k++) {
        memcpy(txt_mat.row(k), &text_emb[k * YW_EMBED_DIM], YW_EMBED_DIM * sizeof(float));
    }
    ex.input(YW_BLOB_TXT, txt_mat);

    ncnn::Mat out_box, out_score;
    ex.extract(YW_BLOB_BOX, out_box);
    ex.extract(YW_BLOB_SCORES, out_score);

    if (out_box.c != 1 || out_score.c != 1) return 0;

    // auto-detect anchor count from box tensor (total elems = 4 * A)
    int total_box = out_box.w * out_box.h;
    int A = total_box / 4;
    if (A <= 0 || total_box != A * 4) return 0;
    box_anchor_first = (out_box.h == A);  // [A,4] => anchors along height

    // scores: [K,A] 或 [A,K], 总元素数 = K * A
    int total_score = out_score.w * out_score.h;
    if (total_score != YW_MAX_CLASSES * A) return 0;
    cls_anchor_first = (out_score.h == YW_MAX_CLASSES);

    std::vector<Object> proposals;
    proposals.reserve(A);
    std::vector<float> box4(4);

    for (int a = 0; a < A; a++) {
        get_box4(out_box, a, box4.data());

        float x0, y0, x1, y1;
        if (YW_BOX_FORMAT == 0) {
            x0 = box4[0]; y0 = box4[1]; x1 = box4[2]; y1 = box4[3];
        } else {
            x0 = box4[0] - box4[2] * 0.5f;
            y0 = box4[1] - box4[3] * 0.5f;
            x1 = box4[0] + box4[2] * 0.5f;
            y1 = box4[1] + box4[3] * 0.5f;
        }

        // 打分已由网络内 BNContrastiveHead + sigmoid 完成, 端侧只取 argmax
        int best_c = 0;
        float prob = 0.f;
        for (int k = 0; k < num_classes; k++) {
            float sc = cls_anchor_first ? out_score.row(k)[a]   // [K,A]
                                        : out_score.row(a)[k];  // [A,K]
            if (sc > prob) { prob = sc; best_c = k; }
        }
        prob *= YW_CONF_SCALE;
        if (prob < prob_threshold) continue;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = best_c;
        obj.prob = prob;
        proposals.push_back(obj);
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.resize(picked.size());
    #pragma omp parallel for num_threads(ncnn::get_big_cpu_count())
    for (int i = 0; i < (int)picked.size(); i++) {
        const Object& src = proposals[picked[i]];
        Object& o = objects[i];
        o = src;

        // de-letterbox: divide by scale (pad is only on right/bottom -> top/left unaffected)
        float bx0 = o.rect.x / scale;
        float by0 = o.rect.y / scale;
        float bx1 = (o.rect.x + o.rect.width) / scale;
        float by1 = (o.rect.y + o.rect.height) / scale;

        bx0 = std::max(std::min(bx0, (float)(img_w - 1)), 0.f);
        by0 = std::max(std::min(by0, (float)(img_h - 1)), 0.f);
        bx1 = std::max(std::min(bx1, (float)(img_w - 1)), 0.f);
        by1 = std::max(std::min(by1, (float)(img_h - 1)), 0.f);

        o.rect.x = bx0;
        o.rect.y = by0;
        o.rect.width = bx1 - bx0;
        o.rect.height = by1 - by0;
    }

    // sort by area (largest first) for stable draw order
    std::sort(objects.begin(), objects.end(),
              [](const Object& a, const Object& b) { return a.rect.area() > b.rect.area(); });

    return 0;
}

// NOTE: drawing used to live here (rect + ASCII label). It moved to the Android
// overlay view (YoloOverlayView) so labels/status text can use the system font
// and therefore render Chinese prompts correctly. Native now only reports boxes.

// JNI bridge for the YOLO-World demo.
// Native side owns the detector + camera render loop. Java supplies the model
// asset, the prompt class names, and their CLIP embeddings (from the text tower).

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <cstdio>

#include <platform.h>
#include <benchmark.h>

#include "yoloworld.h"
#include "ndkcamera.h"

// Centred notice plate. NOTE: the built-in 5x7 font is ASCII-only; non-ASCII
// (e.g. Chinese) prompt/status text falls back to a filled-block glyph, so keep
// on-screen messages short and ASCII where possible.
static void draw_message(YWMat& rgb, const char* msg) {
    const int scale = 2;
    int tw = 0, th = 0;
    yw_text_size(msg, scale, tw, th);
    int plate_w = tw + 8 * scale;
    int plate_h = th + 8 * scale;
    int x = (rgb.cols - plate_w) / 2;
    int y = (rgb.rows - plate_h) / 2;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    yw_fill_rect(rgb, x, y, plate_w, plate_h, 0, 0, 0);
    yw_draw_text(rgb, msg, x + 4 * scale, y + 4 * scale, scale, 255, 255, 255);
}

static void draw_fps(YWMat& rgb, double t) {
    char text[64];
    snprintf(text, sizeof(text), "TIME:%.1f ms", t);
    const int scale = 2;
    int tw = 0, th = 0;
    yw_text_size(text, scale, tw, th);
    int x = rgb.cols - tw - 4 * scale;
    if (x < 0) x = 0;
    yw_fill_rect(rgb, x, 0, tw + 4 * scale, th + 4 * scale, 255, 255, 255);
    yw_draw_text(rgb, text, x + 2 * scale, 2 * scale, scale, 0, 0, 0);
}

static Yoloworld* g_yoloworld = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(YWMat& rgb) const;
};

void MyNdkCamera::on_image_render(YWMat& rgb) const {
    double t = 0.f;
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yoloworld && g_yoloworld->loaded()) {
            if (!g_yoloworld->hasPrompt()) {
                draw_message(rgb, "no prompt: type classes, tap APPLY");
            } else {
                std::vector<Object> objects;
                t = g_yoloworld->detect(rgb, objects);
                g_yoloworld->draw(rgb, objects);
            }
        } else {
            draw_message(rgb, "no model: add yoloworld.param/.bin to assets");
        }
    }
    draw_fps(rgb, t);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    g_camera = new MyNdkCamera;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);
        delete g_yoloworld;
        g_yoloworld = 0;
    }
    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int targetSize, int cpugpu);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_loadModel(JNIEnv* env, jobject thiz,
                                                                             jobject assetManager, jint targetSize, jint cpugpu) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    bool use_gpu = ((int)cpugpu == 1);

    ncnn::MutexLockGuard g(lock);
    if (use_gpu && ncnn::get_gpu_count() == 0) {
        delete g_yoloworld;
        g_yoloworld = 0;
        return JNI_FALSE;
    }
    if (!g_yoloworld) g_yoloworld = new Yoloworld;
    g_yoloworld->load(mgr, use_gpu);
    return JNI_TRUE;
}

// public native boolean setPrompt(String[] names, float[] embeddings);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_setPrompt(JNIEnv* env, jobject thiz,
                                                                            jobjectArray names, jfloatArray embeddings) {
    if (!names || !embeddings) return JNI_FALSE;

    jsize n = env->GetArrayLength(names);
    jsize m = env->GetArrayLength(embeddings);
    if (n <= 0 || m <= 0) return JNI_FALSE;

    std::vector<std::string> name_vec(n);
    for (jsize i = 0; i < n; i++) {
        jstring js = (jstring)env->GetObjectArrayElement(names, i);
        const char* cstr = env->GetStringUTFChars(js, 0);
        name_vec[i] = std::string(cstr ? cstr : "");
        env->ReleaseStringUTFChars(js, cstr);
        env->DeleteLocalRef(js);
    }

    jfloat* emb = env->GetFloatArrayElements(embeddings, 0);
    std::vector<float> emb_vec(emb, emb + m);
    env->ReleaseFloatArrayElements(embeddings, emb, 0);

    ncnn::MutexLockGuard g(lock);
    if (!g_yoloworld) g_yoloworld = new Yoloworld;
    int ret = g_yoloworld->setPrompt(name_vec, emb_vec);
    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_openCamera(JNIEnv* env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1) return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_closeCamera(JNIEnv* env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface) {
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}

}

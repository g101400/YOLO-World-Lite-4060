// JNI bridge for the YOLO-World demo.
//
// Native side owns the detector + camera render loop. Java supplies the model
// asset, the prompt class names and their text embeddings.
//
// Drawing is NOT done here: everything the user sees on top of the preview
// (boxes, labels, status text) is drawn by YoloOverlayView with the Android
// system font, which is the only way to get Chinese labels. Per frame we hand
// the overlay a flat float[]:
//
//   [0] status  0=ok 1=no model 2=no prompt 3=embedding-dim mismatch
//   [1] value   ok -> inference ms ; dim mismatch -> model embedding dim
//   [2] count   number of objects
//   [3..]       per object: x1, y1, x2, y2 (normalised 0..1), label, prob
//
// Boxes are normalised against the *upright* render buffer, so the Java side
// only has to multiply by the view size.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <cstdio>

#include <cstdarg>
#include <cstdlib>

// --- libc++ ABI 兼容桩 -------------------------------------------------------
// ncnn 20260526 预编译库由新版 NDK 构建，其 libc++ 内部断言函数
// std::__ndk1::__libcpp_verbose_abort 在 NDK 21.3 的 libc++ 里不存在。
// 这里按精确修饰名提供定义：仅在 ncnn 内部触发 libc++ 断言(如 vector 越界)时
// 才会被调用 —— 记日志后 abort()。
namespace std {
namespace __ndk1 {
void ncnn_yoloworld_verbose_abort_stub(char const* fmt, ...)
    __asm__("_ZNSt6__ndk122__libcpp_verbose_abortEPKcz");
void ncnn_yoloworld_verbose_abort_stub(char const* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    __android_log_print(ANDROID_LOG_FATAL, "libc++-stub", "%s", buf);
    abort();
}
}
}  // namespace std::__ndk1

#include <platform.h>
#include <benchmark.h>

#include "yoloworld.h"
#include "ndkcamera.h"

enum
{
    YW_STATUS_OK = 0,
    YW_STATUS_NO_MODEL = 1,
    YW_STATUS_NO_PROMPT = 2,
    YW_STATUS_DIM_MISMATCH = 3
};

// ------------------------------------------------------- Java result bridge ---
// The Java listener object is registered from MainActivity.onCreate(), so we
// never need FindClass() from a native camera thread (which has no class
// loader). We only cache a global ref + the method id.

static JavaVM* g_vm = 0;
static jobject g_listener = 0;
static jmethodID g_post_result = 0;
static ncnn::Mutex g_listener_lock;

static void set_listener(JNIEnv* env, jobject listener)
{
    ncnn::MutexLockGuard g(g_listener_lock);
    if (g_listener)
    {
        env->DeleteGlobalRef(g_listener);
        g_listener = 0;
        g_post_result = 0;
    }
    if (!listener) return;

    jclass cls = env->GetObjectClass(listener);
    if (!cls) return;

    jmethodID mid = env->GetMethodID(cls, "postResult", "([F)V");
    env->DeleteLocalRef(cls);
    if (!mid)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "postResult([F)V not found");
        return;
    }

    g_listener = env->NewGlobalRef(listener);
    g_post_result = mid;
}

static void publish_result(const std::vector<float>& data)
{
    if (!g_vm) return;

    JNIEnv* env = 0;
    bool attached = false;
    if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_4) != JNI_OK)
    {
        if (g_vm->AttachCurrentThread(&env, 0) != JNI_OK) return;
        attached = true;
    }

    jobject listener = 0;
    jmethodID mid = 0;
    {
        ncnn::MutexLockGuard g(g_listener_lock);
        listener = g_listener;
        mid = g_post_result;
    }

    if (listener && mid)
    {
        jfloatArray arr = env->NewFloatArray((jsize)data.size());
        if (arr)
        {
            env->SetFloatArrayRegion(arr, 0, (jsize)data.size(), &data[0]);
            env->CallVoidMethod(listener, mid, arr);
            env->DeleteLocalRef(arr);
        }
    }

    if (attached) g_vm->DetachCurrentThread();
}

// ---------------------------------------------------------------- detector ---

static Yoloworld* g_yoloworld = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(YWMat& rgb) const;
};

void MyNdkCamera::on_image_render(YWMat& rgb) const
{
    std::vector<float> out;

    {
        ncnn::MutexLockGuard g(lock);

        if (!g_yoloworld || !g_yoloworld->loaded())
        {
            out.push_back((float)YW_STATUS_NO_MODEL);
            out.push_back(0.f);
            out.push_back(0.f);
        }
        else if (!g_yoloworld->hasPrompt())
        {
            out.push_back((float)YW_STATUS_NO_PROMPT);
            out.push_back(0.f);
            out.push_back(0.f);
        }
        else
        {
            const double t0 = ncnn::get_current_time();
            std::vector<Object> objects;
            g_yoloworld->detect(rgb, objects);
            const double t = ncnn::get_current_time() - t0;

            if (g_yoloworld->dimMismatch())
            {
                out.push_back((float)YW_STATUS_DIM_MISMATCH);
                out.push_back((float)g_yoloworld->modelEmbedDim());
                out.push_back(0.f);
            }
            else
            {
                const float iw = (float)(rgb.cols > 0 ? rgb.cols : 1);
                const float ih = (float)(rgb.rows > 0 ? rgb.rows : 1);

                out.push_back((float)YW_STATUS_OK);
                out.push_back((float)t);
                out.push_back((float)objects.size());

                for (size_t i = 0; i < objects.size(); i++)
                {
                    const Object& o = objects[i];
                    float x1 = o.rect.x / iw;
                    float y1 = o.rect.y / ih;
                    float x2 = (o.rect.x + o.rect.width) / iw;
                    float y2 = (o.rect.y + o.rect.height) / ih;
                    if (x1 < 0.f) x1 = 0.f;
                    if (y1 < 0.f) y1 = 0.f;
                    if (x2 > 1.f) x2 = 1.f;
                    if (y2 > 1.f) y2 = 1.f;
                    out.push_back(x1);
                    out.push_back(y1);
                    out.push_back(x2);
                    out.push_back(y2);
                    out.push_back((float)o.label);
                    out.push_back(o.prob);
                }
            }
        }
    }

    publish_result(out);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    g_vm = vm;
    g_camera = new MyNdkCamera;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);
        delete g_yoloworld;
        g_yoloworld = 0;
    }
    delete g_camera;
    g_camera = 0;
}

// public static native void setResultListener(Object listener);
JNIEXPORT void JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_setResultListener(JNIEnv* env, jclass clazz,
                                                                               jobject listener)
{
    set_listener(env, listener);
}

// public native boolean loadModel(AssetManager mgr, int targetSize, int cpugpu);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_loadModel(JNIEnv* env, jobject thiz,
                                                                             jobject assetManager, jint targetSize, jint cpugpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    bool use_gpu = false;  // CPU-only ncnn build; GPU path needs the vulkan package
    (void)cpugpu;

    ncnn::MutexLockGuard g(lock);
    if (!g_yoloworld) g_yoloworld = new Yoloworld;
    int ret = g_yoloworld->load(mgr, use_gpu);
    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native boolean setPrompt(String[] names, float[] embeddings);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_setPrompt(JNIEnv* env, jobject thiz,
                                                                            jobjectArray names, jfloatArray embeddings)
{
    if (!names || !embeddings) return JNI_FALSE;

    jsize n = env->GetArrayLength(names);
    jsize m = env->GetArrayLength(embeddings);
    if (n <= 0 || m <= 0) return JNI_FALSE;

    std::vector<std::string> name_vec(n);
    for (jsize i = 0; i < n; i++)
    {
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
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1) return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_ncnn_yoloworld_demo_NcnnYoloworld_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}

} // extern "C"

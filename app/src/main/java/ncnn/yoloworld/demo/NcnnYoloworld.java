package ncnn.yoloworld.demo;

import android.content.res.AssetManager;
import android.view.Surface;

public class NcnnYoloworld {
    // load YOLO-World ncnn model from assets (yoloworld.param / yoloworld.bin)
    public native boolean loadModel(AssetManager mgr, int targetSize, int cpugpu);

    // set open-vocabulary prompt: class names + their text embeddings (flattened)
    public native boolean setPrompt(String[] names, float[] embeddings);

    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);

    /**
     * 注册结果监听对象：原生渲染线程会对它调用
     * {@code postResult(float[])}。传 null 取消注册。
     */
    public static native void setResultListener(Object listener);

    static {
        System.loadLibrary("ncnn_yoloworld");
    }
}

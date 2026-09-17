package ncnn.yoloworld.demo;

import android.content.res.AssetManager;
import android.view.Surface;

public class NcnnYoloworld {
    // load YOLO-World ncnn model from assets (yoloworld.param / yoloworld.bin)
    public native boolean loadModel(AssetManager mgr, int targetSize, int cpugpu);

    // set open-vocabulary prompt: class names + their CLIP embeddings (flattened)
    public native boolean setPrompt(String[] names, float[] embeddings);

    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);

    static {
        System.loadLibrary("ncnn_yoloworld");
    }
}

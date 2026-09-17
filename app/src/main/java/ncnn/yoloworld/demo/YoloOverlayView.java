package ncnn.yoloworld.demo;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/**
 * 透明覆盖层：在相机预览之上绘制检测框、类别标签和状态提示。
 *
 * <p>原先这些内容由原生 5x7 点阵字绘制，只能显示 ASCII，中文会变成方块。改成
 * Android 侧绘制后可以直接使用系统字体，因此中文提示词可以正常显示。
 *
 * <p>{@link #postResult(float[])} 由原生渲染线程调用。
 */
public class YoloOverlayView extends View {

    /** 嵌入维度不匹配时回调（已切到主线程）。 */
    public interface Listener {
        void onEmbedDimMismatch(int modelDim);
    }

    public static final int STATUS_OK = 0;
    public static final int STATUS_NO_MODEL = 1;
    public static final int STATUS_NO_PROMPT = 2;
    public static final int STATUS_DIM_MISMATCH = 3;

    private static final int[] COLORS = {
            0xFF36F043, 0xFFE91E63, 0xFF9C27B0, 0xFF673AB7, 0xFF3F51B5,
            0xFF2196F3, 0xFF00BCD4, 0xFF009688, 0xFF4CAF50, 0xFF8BC34A,
            0xFFFFC107, 0xFFFF9800, 0xFFFF5722, 0xFF795548, 0xFF607D8B,
            0xFFE040FB, 0xFF18FFFF, 0xFFFFD740, 0xFFB2FF59
    };

    private final Object lock = new Object();
    private float[] pending;

    private final Paint boxPaint = new Paint();
    private final Paint platePaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint hintTextPaint = new Paint();
    private final Paint hintPlatePaint = new Paint();
    private final RectF rect = new RectF();

    private String[] classNames = new String[0];
    private Listener listener;
    private int lastNotifiedDim = 0;

    private final float density;
    private final float labelTextSize;

    public YoloOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

        density = getResources().getDisplayMetrics().density;
        labelTextSize = 14f * density;

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(3f, 2f * density));
        boxPaint.setAntiAlias(true);

        platePaint.setStyle(Paint.Style.FILL);
        platePaint.setAntiAlias(true);

        textPaint.setAntiAlias(true);
        textPaint.setTextSize(labelTextSize);
        textPaint.setFakeBoldText(true);

        hintTextPaint.setAntiAlias(true);
        hintTextPaint.setTextSize(15f * density);
        hintTextPaint.setColor(Color.WHITE);
        hintTextPaint.setFakeBoldText(true);

        hintPlatePaint.setAntiAlias(true);
        hintPlatePaint.setColor(0xC0000000);

        setWillNotDraw(false);
    }

    public void setClassNames(String[] names) {
        classNames = (names == null) ? new String[0] : names;
    }

    public void setListener(Listener l) {
        listener = l;
    }

    /** 由原生渲染线程调用（约每帧一次）。 */
    public void postResult(float[] data) {
        if (data == null || data.length < 3) return;
        synchronized (lock) {
            pending = data;
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        float[] d;
        synchronized (lock) {
            d = pending;
        }
        if (d == null || d.length < 3) return;

        final int status = (int) d[0];
        final float value = d[1];

        if (status == STATUS_NO_MODEL) {
            drawHint(canvas, getContext().getString(R.string.ov_no_model));
            return;
        }
        if (status == STATUS_NO_PROMPT) {
            drawHint(canvas, getContext().getString(R.string.ov_no_prompt));
            return;
        }
        if (status == STATUS_DIM_MISMATCH) {
            final int modelDim = (int) value;
            drawHint(canvas, getContext().getString(R.string.ov_dim_mismatch, modelDim));
            if (listener != null && modelDim > 0 && modelDim != lastNotifiedDim) {
                lastNotifiedDim = modelDim;
                post(new Runnable() {
                    @Override
                    public void run() {
                        listener.onEmbedDimMismatch(modelDim);
                    }
                });
            }
            return;
        }

        final int count = (int) d[2];
        final float w = getWidth();
        final float h = getHeight();

        for (int i = 0; i < count; i++) {
            final int base = 3 + i * 6;
            if (base + 5 >= d.length) break;

            final float x1 = d[base] * w;
            final float y1 = d[base + 1] * h;
            final float x2 = d[base + 2] * w;
            final float y2 = d[base + 3] * h;
            final int label = (int) d[base + 4];
            final float prob = d[base + 5];

            if (x2 - x1 < 1f || y2 - y1 < 1f) continue;

            final int color = COLORS[i % COLORS.length];

            boxPaint.setColor(color);
            rect.set(x1, y1, x2, y2);
            canvas.drawRoundRect(rect, 4f * density, 4f * density, boxPaint);

            final String name = (label >= 0 && label < classNames.length && classNames[label] != null
                    && classNames[label].length() > 0) ? classNames[label] : String.valueOf(label);
            final String text = getContext().getString(R.string.ov_label, name, prob * 100f);

            final float tw = textPaint.measureText(text);
            final float th = labelTextSize * 1.35f;

            float lx = x1;
            float ly = y1 - th - 2f * density;
            if (ly < 0f) ly = y1 + 2f * density;                 // 顶部放不下则放到框内
            if (lx + tw + 6f * density > w) lx = w - tw - 6f * density;
            if (lx < 0f) lx = 0f;
            if (ly + th > h) ly = h - th;

            platePaint.setColor(color);
            rect.set(lx, ly, lx + tw + 6f * density, ly + th);
            canvas.drawRoundRect(rect, 3f * density, 3f * density, platePaint);

            // 亮色底牌配深色字，保证可读性
            final int sum = Color.red(color) + Color.green(color) + Color.blue(color);
            textPaint.setColor(sum >= 381 ? Color.BLACK : Color.WHITE);

            final float baseline = ly + th - 2f * density;
            canvas.drawText(text, lx + 3f * density, baseline, textPaint);
        }

        drawTiming(canvas, value, w);
    }

    private void drawTiming(Canvas canvas, float ms, float w) {
        final String text = getContext().getString(R.string.ov_time, ms);
        final float tw = hintTextPaint.measureText(text);
        final float th = hintTextPaint.getTextSize() * 1.4f;
        final float x = w - tw - 6f * density;
        final float y = 4f * density;

        rect.set(x - 3f * density, y, x + tw + 3f * density, y + th);
        canvas.drawRoundRect(rect, 3f * density, 3f * density, hintPlatePaint);
        canvas.drawText(text, x, y + th - 4f * density, hintTextPaint);
    }

    private void drawHint(Canvas canvas, String msg) {
        final float w = getWidth();
        final float h = getHeight();
        final float tw = hintTextPaint.measureText(msg);
        final float th = hintTextPaint.getTextSize() * 1.5f;

        final float x = Math.max(4f * density, (w - tw) / 2f);
        final float y = (h - th) / 2f;

        rect.set(x - 10f * density, y, x + tw + 10f * density, y + th);
        canvas.drawRoundRect(rect, 6f * density, 6f * density, hintPlatePaint);
        canvas.drawText(msg, x, y + th - 6f * density, hintTextPaint);
    }
}

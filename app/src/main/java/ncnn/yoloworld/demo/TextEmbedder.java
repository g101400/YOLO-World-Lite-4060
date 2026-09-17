package ncnn.yoloworld.demo;

import android.content.res.AssetManager;
import android.util.Log;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;

/**
 * 端侧文本 → 嵌入向量，App 自身完成，不再依赖 PC 上的文本塔服务。
 *
 * <p>两级来源，优先使用第一级：
 * <ol>
 *   <li>{@code assets/prompt_emb.bin} —— 预计算的 CLIP 文本嵌入词表（真实文本塔
 *       对齐，词表固定）。用 {@code tools/build_prompt_table.py} 生成后放进 assets。</li>
 *   <li>内置离线编码器 —— 字符 / 二元组的确定性哈希投影。任意提示词可用、不需要
 *       任何服务，但它<b>不是</b> CLIP 对齐的，只有配合随包一起提供的链路验证模型
 *       才有意义；真实权重请用第 1 级或远端服务。</li>
 * </ol>
 *
 * <p>词表文件格式（小端序）：
 * <pre>
 *   char[4] magic = 'Y','W','E','M'
 *   uint32  version = 1
 *   uint32  count
 *   uint32  dim
 *   repeat count:
 *     uint32 nameLen
 *     byte[nameLen] name (UTF-8)
 *     float32[dim]  embedding (已 L2 归一化)
 * </pre>
 */
public final class TextEmbedder {

    private static final String TAG = "TextEmbedder";
    private static final String TABLE_ASSET = "prompt_emb.bin";

    /** CLIP ViT-B/32 与随包模型的维度，也是默认值。 */
    public static final int DEFAULT_DIM = 512;

    private static HashMap<String, float[]> table;
    private static int tableDim;

    private TextEmbedder() {
    }

    /** 读取 assets 里的嵌入词表（不存在就静默跳过，使用离线编码器）。 */
    public static synchronized void loadTable(AssetManager mgr) {
        if (table != null) return;
        table = new HashMap<String, float[]>();
        tableDim = 0;

        InputStream raw = null;
        try {
            raw = mgr.open(TABLE_ASSET);
        } catch (IOException e) {
            Log.i(TAG, TABLE_ASSET + " 不存在，使用内置离线编码器");
            return;
        }

        DataInputStream in = new DataInputStream(raw);
        try {
            byte[] magic = new byte[4];
            in.readFully(magic);
            if (magic[0] != 'Y' || magic[1] != 'W' || magic[2] != 'E' || magic[3] != 'M') {
                throw new IOException("bad magic");
            }
            int version = readIntLE(in);
            if (version != 1) throw new IOException("unsupported version " + version);

            int count = readIntLE(in);
            int dim = readIntLE(in);
            if (count <= 0 || dim <= 0) throw new IOException("empty table");

            for (int i = 0; i < count; i++) {
                int nameLen = readIntLE(in);
                if (nameLen <= 0 || nameLen > 4096) throw new IOException("bad name length");
                byte[] nb = new byte[nameLen];
                in.readFully(nb);
                String name = new String(nb, "UTF-8");

                float[] v = new float[dim];
                for (int j = 0; j < dim; j++) v[j] = readFloatLE(in);
                table.put(name.trim(), v);
            }
            tableDim = dim;
            Log.i(TAG, "已加载嵌入词表：" + table.size() + " 条, dim=" + dim);
        } catch (IOException e) {
            Log.w(TAG, "嵌入词表读取失败，改用离线编码器: " + e.getMessage());
            table = new HashMap<String, float[]>();
            tableDim = 0;
        } finally {
            try {
                in.close();
            } catch (IOException ignored) {
            }
        }
    }

    public static synchronized boolean hasTable() {
        return table != null && !table.isEmpty();
    }

    public static synchronized int tableSize() {
        return table == null ? 0 : table.size();
    }

    private static synchronized float[] lookup(String name) {
        if (table == null || table.isEmpty() || name == null) return null;
        float[] v = table.get(name.trim());
        if (v == null || v.length == 0) return null;
        // 返回副本，避免调用方修改共享词表
        float[] copy = new float[v.length];
        System.arraycopy(v, 0, copy, 0, v.length);
        return copy;
    }

    /** 批量编码，返回按行拼接的扁平数组（N * dim）。 */
    public static float[] encodeAll(List<String> names, int dim) {
        int n = names.size();
        float[] out = new float[n * dim];
        for (int i = 0; i < n; i++) {
            float[] v = encode(names.get(i), dim);
            System.arraycopy(v, 0, out, i * dim, dim);
        }
        return out;
    }

    /** 先查词表，维度不符或未命中则退回离线编码器。 */
    public static float[] encode(String name, int dim) {
        float[] v = lookup(name);
        if (v != null && v.length == dim) return v;
        if (v != null) {
            Log.i(TAG, "词表维度(" + v.length + ")与模型(" + dim + ")不一致，改用离线编码器");
        }
        return encodeLocal(name, dim);
    }

    /**
     * 内置离线编码器：对字符串做「单字 + 相邻二元组」特征哈希，投影到 dim 维并
     * L2 归一化。确定性、无依赖、支持中文；相邻字符共享特征，所以「汽车」和
     * 「车」会有一定相似度。它不是 CLIP 对齐的语义空间。
     */
    public static float[] encodeLocal(String text, int dim) {
        if (dim <= 0) dim = DEFAULT_DIM;
        float[] v = new float[dim];

        String t = (text == null) ? "" : text.trim();
        if (t.length() == 0) t = " ";

        for (int i = 0; i < t.length(); i++) {
            addFeature(v, t.substring(i, i + 1));
        }
        for (int i = 0; i + 1 < t.length(); i++) {
            addFeature(v, t.substring(i, i + 2));
        }

        double norm = 0.0;
        for (int i = 0; i < dim; i++) norm += (double) v[i] * v[i];
        norm = Math.sqrt(norm);
        if (norm > 1e-8) {
            float inv = (float) (1.0 / norm);
            for (int i = 0; i < dim; i++) v[i] *= inv;
        } else {
            v[0] = 1.f;
        }
        return v;
    }

    private static void addFeature(float[] v, String token) {
        long h = fnv1a(token);
        int idx = (int) ((h >>> 1) % v.length);
        v[idx] += ((h & 1L) == 0L) ? 1.f : -1.f;

        long h2 = fnv1a(token + "#");
        int idx2 = (int) ((h2 >>> 1) % v.length);
        v[idx2] += ((h2 & 1L) == 0L) ? 0.5f : -0.5f;
    }

    private static long fnv1a(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= (long) s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h;
    }

    // ------- little-endian helpers -------

    private static int readIntLE(InputStream in) throws IOException {
        int b0 = in.read();
        int b1 = in.read();
        int b2 = in.read();
        int b3 = in.read();
        if ((b0 | b1 | b2 | b3) < 0) throw new IOException("unexpected eof");
        return (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
    }

    private static float readFloatLE(InputStream in) throws IOException {
        return Float.intBitsToFloat(readIntLE(in));
    }
}

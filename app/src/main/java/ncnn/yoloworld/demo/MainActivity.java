package ncnn.yoloworld.demo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;

public class MainActivity extends Activity implements SurfaceHolder.Callback,
        YoloOverlayView.Listener {

    private static final String TAG = "MainActivity";
    public static final int REQUEST_CAMERA = 100;

    private final NcnnYoloworld ncnnyoloworld = new NcnnYoloworld();

    private int facing = 1;              // 0 = 前置摄像头, 1 = 后置摄像头
    private int current_cpugpu = 0;
    private boolean camera_opened = false;

    private SurfaceView cameraView;
    private YoloOverlayView overlayView;
    private EditText editPrompt;
    private EditText editServer;
    private CheckBox checkUseServer;
    private TextView textStatus;

    // 维度自愈：模型实际嵌入维度与提示词不一致时，用模型维度重算一次
    private int lastDimTried = 0;
    private int dimRetryCount = 0;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        overlayView = (YoloOverlayView) findViewById(R.id.overlay);
        overlayView.setListener(this);

        editPrompt = (EditText) findViewById(R.id.editPrompt);
        editServer = (EditText) findViewById(R.id.editServer);
        checkUseServer = (CheckBox) findViewById(R.id.checkUseServer);
        textStatus = (TextView) findViewById(R.id.textStatus);

        NcnnYoloworld.setResultListener(overlayView);
        TextEmbedder.loadTable(getAssets());

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                facing = 1 - facing;
                closeCamera();
                openCamera();
            }
        });

        Spinner spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        ArrayAdapter<CharSequence> gpuAdapter = ArrayAdapter.createFromResource(
                this, R.array.cpugpu_array, android.R.layout.simple_spinner_item);
        gpuAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerCPUGPU.setAdapter(gpuAdapter);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Button buttonApplyPrompt = (Button) findViewById(R.id.buttonApplyPrompt);
        buttonApplyPrompt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                applyPrompt(TextEmbedder.DEFAULT_DIM, checkUseServer.isChecked());
            }
        });

        requestCameraIfNeeded();
        reload();
        applyPrompt(TextEmbedder.DEFAULT_DIM, false);
    }

    private void reload() {
        boolean ret = ncnnyoloworld.loadModel(getAssets(), 640, current_cpugpu);
        if (!ret) {
            Log.e(TAG, "loadModel failed");
            textStatus.setText(current_cpugpu == 1 ? R.string.status_gpu_unavailable
                    : R.string.status_model_fail);
        } else {
            textStatus.setText(TextEmbedder.hasTable()
                    ? getString(R.string.status_ready_table, TextEmbedder.tableSize())
                    : getString(R.string.status_ready_local));
        }
    }

    // ------------------------------------------------------------ 提示词处理 ----

    private void applyPrompt(final int dim, final boolean useServer) {
        final String raw = editPrompt.getText().toString();
        if (raw.trim().length() == 0) {
            Toast.makeText(this, R.string.prompt_empty, Toast.LENGTH_SHORT).show();
            return;
        }

        // 中英文逗号、顿号、分号、换行都可以作分隔符
        String[] parts = raw.split("[,，、;；\n]");
        final ArrayList<String> names = new ArrayList<String>();
        for (int i = 0; i < parts.length; i++) {
            String s = parts[i].trim();
            if (s.length() > 0) names.add(s);
        }
        if (names.isEmpty()) {
            Toast.makeText(this, R.string.prompt_empty, Toast.LENGTH_SHORT).show();
            return;
        }

        lastDimTried = dim;
        final String serverUrl = editServer.getText().toString().trim();
        if (useServer) {
            textStatus.setText(R.string.status_fetching);
        } else {
            textStatus.setText(getString(R.string.status_encoding, names.size(), dim));
        }

        new Thread(new Runnable() {
            @Override
            public void run() {
                final ArrayList<String> finalNames = names;
                float[] emb = null;
                boolean usedServer = useServer;

                if (useServer) {
                    emb = fetchEmbeddings(serverUrl, finalNames);
                }
                if (emb == null) {
                    if (useServer) usedServer = false;   // 服务不可用 -> 退回端侧编码
                    emb = TextEmbedder.encodeAll(finalNames, dim);
                }
                if (emb == null) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            textStatus.setText(R.string.status_set_fail);
                        }
                    });
                    return;
                }

                final float[] finalEmb = emb;
                final boolean finalUsedServer = usedServer;
                String[] nameArr = new String[finalNames.size()];
                finalNames.toArray(nameArr);
                final boolean ok = ncnnyoloworld.setPrompt(nameArr, finalEmb);
                final int n = finalNames.size();

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        overlayView.setClassNames(toArray(finalNames));
                        if (!ok) {
                            textStatus.setText(R.string.status_set_fail);
                            return;
                        }
                        String source = finalUsedServer ? getString(R.string.source_server)
                                : (TextEmbedder.hasTable() ? getString(R.string.source_table)
                                : getString(R.string.source_local));
                        String msg = getString(R.string.status_applied, n, source);
                        if (useServer && !finalUsedServer) {
                            msg = msg + getString(R.string.status_server_fallback);
                        }
                        textStatus.setText(msg);
                        Toast.makeText(MainActivity.this, msg, Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }).start();
    }

    private static String[] toArray(ArrayList<String> list) {
        String[] arr = new String[list.size()];
        list.toArray(arr);
        return arr;
    }

    /** 模型期望的嵌入维度与提示词不一致：改用端侧编码并自动重算。 */
    @Override
    public void onEmbedDimMismatch(int modelDim) {
        if (modelDim <= 0 || modelDim == lastDimTried) return;
        if (checkUseServer.isChecked()) {
            textStatus.setText(getString(R.string.status_dim_server, modelDim));
            return;
        }
        if (dimRetryCount >= 3) {
            textStatus.setText(getString(R.string.status_dim_giveup, modelDim));
            return;
        }
        dimRetryCount++;
        textStatus.setText(getString(R.string.status_dim_retry, modelDim));
        applyPrompt(modelDim, false);
    }

    /** 远端文本编码服务（可选）。POST {"prompts":[...]} -> {"embeddings":[[...]]} */
    private float[] fetchEmbeddings(String server, ArrayList<String> names) {
        if (server == null || server.length() == 0) return null;
        if (!server.startsWith("http://") && !server.startsWith("https://")) {
            server = "http://" + server;
        }
        try {
            JSONObject req = new JSONObject();
            JSONArray arr = new JSONArray();
            for (int i = 0; i < names.size(); i++) arr.put(names.get(i));
            req.put("prompts", arr);

            URL url = new URL(server + "/embed");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            conn.setDoOutput(true);
            conn.setConnectTimeout(4000);
            conn.setReadTimeout(15000);
            OutputStream os = conn.getOutputStream();
            os.write(req.toString().getBytes("UTF-8"));
            os.close();

            int code = conn.getResponseCode();
            InputStream is = (code == 200) ? conn.getInputStream() : conn.getErrorStream();
            StringBuilder sb = new StringBuilder();
            if (is != null) {
                BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                String line;
                while ((line = br.readLine()) != null) sb.append(line);
                br.close();
            }
            conn.disconnect();

            if (code != 200) {
                Log.e(TAG, "embed http " + code + " " + sb.toString());
                return null;
            }

            JSONObject res = new JSONObject(sb.toString());
            JSONArray embs = res.getJSONArray("embeddings");
            int rows = embs.length();
            if (rows <= 0) return null;
            int cols = embs.getJSONArray(0).length();
            float[] flat = new float[rows * cols];
            int k = 0;
            for (int i = 0; i < rows; i++) {
                JSONArray row = embs.getJSONArray(i);
                for (int j = 0; j < cols; j++) flat[k++] = (float) row.getDouble(j);
            }
            return flat;
        } catch (Exception e) {
            Log.e(TAG, "fetchEmbeddings " + e.getMessage());
            return null;
        }
    }

    // -------------------------------------------------------------- 摄像头 ----

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraIfNeeded() {
        if (!hasCameraPermission()) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_CAMERA) return;
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            textStatus.setText(R.string.status_no_camera_perm);
        }
    }

    private void openCamera() {
        if (!hasCameraPermission()) return;
        ncnnyoloworld.openCamera(facing);
        camera_opened = true;
    }

    private void closeCamera() {
        if (!camera_opened) return;
        ncnnyoloworld.closeCamera();
        camera_opened = false;
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        ncnnyoloworld.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    @Override
    public void onResume() {
        super.onResume();
        requestCameraIfNeeded();
        if (hasCameraPermission()) openCamera();
    }

    @Override
    public void onPause() {
        super.onPause();
        closeCamera();
    }
}

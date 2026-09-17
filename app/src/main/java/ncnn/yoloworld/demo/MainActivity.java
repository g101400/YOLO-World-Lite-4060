package ncnn.yoloworld.demo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.text.method.LinkMovementMethod;
import android.text.util.Linkify;
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

    // ---- 在线升级（GitHub Release 为主渠道，百度网盘为备用） ----
    private static final String REPO_PAGE =
            "https://github.com/g101400/YOLO-World-Lite-4060";
    private static final String VERSION_URL =
            "https://raw.githubusercontent.com/g101400/YOLO-World-Lite-4060/main/update/version.json";

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

        Button buttonMenu = (Button) findViewById(R.id.buttonMenu);
        buttonMenu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                showMainMenu();
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
        if (names.size() > 10) {
            Toast.makeText(this, R.string.prompt_too_many, Toast.LENGTH_SHORT).show();
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

    // ------------------------------------------------------------ 主菜单 ----

    private void showMainMenu() {
        final String[] items = getResources().getStringArray(R.array.main_menu);
        new AlertDialog.Builder(this)
                .setTitle(R.string.menu_title)
                .setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        switch (which) {
                            case 0: showDoc(R.string.doc_compare); break;
                            case 1: showDoc(R.string.doc_industry); break;
                            case 2: showDoc(R.string.doc_llm); break;
                            case 3: showUpdateMenu(); break;
                            case 4:
                                showDoc(getString(R.string.doc_expand, REPO_PAGE));
                                break;
                            case 5: showDoc(R.string.doc_help); break;
                            case 6:
                                showDoc(getString(R.string.doc_about,
                                        REPO_PAGE, getAppVersionName(), getAppVersionCode()));
                                break;
                        }
                    }
                })
                .setNegativeButton(R.string.update_btn_close, null)
                .show();
    }

    /** 显示文档对话框（可滚动，URL 自动可点）。 */
    private void showDoc(int textResId) {
        showDoc(getString(textResId));
    }

    private void showDoc(String text) {
        android.widget.ScrollView scroll = new android.widget.ScrollView(this);
        TextView tv = new TextView(this);
        tv.setPadding(48, 32, 48, 48);
        tv.setTextSize(14);
        tv.setText(text);
        tv.setAutoLinkMask(Linkify.WEB_URLS);
        tv.setMovementMethod(LinkMovementMethod.getInstance());
        tv.setTextIsSelectable(true);
        scroll.addView(tv);

        new AlertDialog.Builder(this)
                .setView(scroll)
                .setPositiveButton(R.string.update_btn_close, null)
                .show();
    }

    // ---------------------------------------------------------- 软件升级 ----

    private void showUpdateMenu() {
        final String[] items = {
                "检查更新",
                getString(R.string.update_channel)
        };
        new AlertDialog.Builder(this)
                .setTitle(R.string.update_menu_title)
                .setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (which == 0) checkUpdate();
                        else showDoc(getString(R.string.doc_update_channel, REPO_PAGE));
                    }
                })
                .setNegativeButton(R.string.update_btn_close, null)
                .show();
    }

    private void checkUpdate() {
        textStatus.setText(R.string.update_checking);
        final String localVersion = getAppVersionName();
        final int localCode = getAppVersionCode();
        new Thread(new Runnable() {
            @Override
            public void run() {
                String remoteName = null;
                int remoteCode = 0;
                String notes = "";
                String apkUrl = "";
                String raw = httpGet(VERSION_URL, 6000);
                if (raw != null) {
                    try {
                        JSONObject j = new JSONObject(raw);
                        remoteName = j.optString("versionName", null);
                        remoteCode = j.optInt("versionCode", 0);
                        notes = j.optString("notes", "");
                        apkUrl = j.optString("apk", REPO_PAGE + "/releases");
                    } catch (Exception e) {
                        Log.e(TAG, "checkUpdate parse " + e.getMessage());
                    }
                }
                final boolean hasNew = remoteName != null && remoteCode > localCode;
                final String fRemote = remoteName;
                final String fNotes = notes;
                final String fApkUrl = apkUrl;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (fRemote == null) {
                            showDoc(getString(R.string.update_check_fail, REPO_PAGE + "/releases"));
                            return;
                        }
                        if (hasNew) {
                            showUpdateDialog(fRemote, localVersion, fNotes, fApkUrl);
                        } else {
                            showDoc(getString(R.string.update_already_latest, localVersion));
                        }
                    }
                });
            }
        }).start();
    }

    private void showUpdateDialog(final String remote, String local,
                                  String notes, final String apkUrl) {
        String msg = getString(R.string.update_new_version, remote, local,
                TextUtils.isEmpty(notes) ? "" : "\n更新内容：\n" + notes);
        new AlertDialog.Builder(this)
                .setTitle(getString(R.string.btn_menu) + " · " + getString(R.string.update_checking).replace("…", ""))
                .setMessage(msg)
                .setPositiveButton(R.string.update_btn_download, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        openBrowser(apkUrl);
                    }
                })
                .setNeutralButton(R.string.update_btn_open_page, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        openBrowser(REPO_PAGE + "/releases");
                    }
                })
                .setNegativeButton(R.string.update_btn_close, null)
                .show();
    }

    private void openBrowser(String url) {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
        } catch (Exception e) {
            Toast.makeText(this, "无法打开浏览器：" + url, Toast.LENGTH_LONG).show();
        }
    }

    private String getAppVersionName() {
        try {
            PackageInfo pi = getPackageManager().getPackageInfo(getPackageName(), 0);
            return pi.versionName;
        } catch (Exception e) {
            return "unknown";
        }
    }

    private int getAppVersionCode() {
        try {
            PackageInfo pi = getPackageManager().getPackageInfo(getPackageName(), 0);
            return pi.versionCode;
        } catch (Exception e) {
            return 0;
        }
    }

    /** 简单 GET，返回响应体文本或 null。 */
    private static String httpGet(String urlStr, int timeoutMs) {
        try {
            HttpURLConnection conn = (HttpURLConnection) new URL(urlStr).openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(timeoutMs);
            conn.setReadTimeout(timeoutMs);
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
            return (code == 200) ? sb.toString() : null;
        } catch (Exception e) {
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

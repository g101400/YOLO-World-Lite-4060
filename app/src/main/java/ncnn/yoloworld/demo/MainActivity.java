package ncnn.yoloworld.demo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;

    private NcnnYoloworld ncnnyoloworld = new NcnnYoloworld();
    private int facing = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    private EditText editPrompt;
    private EditText editServer;
    private TextView textStatus;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        editPrompt = (EditText) findViewById(R.id.editPrompt);
        editServer = (EditText) findViewById(R.id.editServer);
        textStatus = (TextView) findViewById(R.id.textStatus);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                int new_facing = 1 - facing;
                ncnnyoloworld.closeCamera();
                ncnnyoloworld.openCamera(new_facing);
                facing = new_facing;
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
            public void onNothingSelected(AdapterView<?> arg0) { }
        });

        Button buttonApplyPrompt = (Button) findViewById(R.id.buttonApplyPrompt);
        buttonApplyPrompt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                applyPrompt();
            }
        });

        reload();
        applyPrompt();
    }

    private void reload() {
        boolean ret = ncnnyoloworld.loadModel(getAssets(), 640, current_cpugpu);
        if (!ret) {
            Log.e("MainActivity", "loadModel failed");
            textStatus.setText(R.string.status_model_fail);
        } else {
            textStatus.setText(R.string.status_wait_prompt);
        }
    }

    private void applyPrompt() {
        final String raw = editPrompt.getText().toString();
        final String server = editServer.getText().toString().trim();
        if (raw.trim().isEmpty()) {
            Toast.makeText(this, R.string.prompt_empty, Toast.LENGTH_SHORT).show();
            return;
        }
        // split by English/Chinese comma or newline
        String[] parts = raw.split("[,，\n]");
        final ArrayList<String> names = new ArrayList<String>();
        for (int i = 0; i < parts.length; i++) {
            String s = parts[i].trim();
            if (!s.isEmpty()) names.add(s);
        }
        if (names.isEmpty()) {
            Toast.makeText(this, R.string.prompt_empty, Toast.LENGTH_SHORT).show();
            return;
        }

        textStatus.setText(R.string.status_fetching);
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    final float[] emb = fetchEmbeddings(server, names);
                    if (emb == null) {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                textStatus.setText(R.string.status_embed_fail);
                                Toast.makeText(MainActivity.this, R.string.toast_req_fail, Toast.LENGTH_LONG).show();
                            }
                        });
                        return;
                    }
                    String[] nameArr = new String[names.size()];
                    names.toArray(nameArr);
                    final boolean ok = ncnnyoloworld.setPrompt(nameArr, emb);
                    final int n = names.size();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (ok) {
                                textStatus.setText(String.format(getString(R.string.status_applied), n));
                                Toast.makeText(MainActivity.this,
                                        String.format(getString(R.string.toast_applied), n),
                                        Toast.LENGTH_SHORT).show();
                            } else {
                                textStatus.setText(R.string.status_set_fail);
                            }
                        }
                    });
                } catch (final Exception e) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            textStatus.setText(getString(R.string.status_error) + e.getMessage());
                            Toast.makeText(MainActivity.this,
                                    getString(R.string.status_error) + e.getMessage(),
                                    Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        }).start();
    }

    // POST {"prompts":[...]} -> {"dim":E,"embeddings":[[...]*N]} ; returns flattened N*E floats
    private float[] fetchEmbeddings(String server, ArrayList<String> names) {
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
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(20000);
            OutputStream os = conn.getOutputStream();
            os.write(req.toString().getBytes("UTF-8"));
            os.close();

            int code = conn.getResponseCode();
            InputStream is = (code == 200) ? conn.getInputStream() : conn.getErrorStream();
            StringBuilder sb = new StringBuilder();
            BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
            br.close();
            conn.disconnect();

            if (code != 200) {
                Log.e("MainActivity", "embed http " + code + " " + sb.toString());
                return null;
            }

            JSONObject res = new JSONObject(sb.toString());
            JSONArray embs = res.getJSONArray("embeddings");
            int rows = embs.length();
            int cols = embs.getJSONArray(0).length();
            float[] flat = new float[rows * cols];
            int k = 0;
            for (int i = 0; i < rows; i++) {
                JSONArray row = embs.getJSONArray(i);
                for (int j = 0; j < cols; j++) {
                    flat[k++] = (float) row.getDouble(j);
                }
            }
            return flat;
        } catch (Exception e) {
            Log.e("MainActivity", "fetchEmbeddings " + e.getMessage());
            return null;
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        ncnnyoloworld.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) { }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) { }

    @Override
    public void onResume() {
        super.onResume();
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
        ncnnyoloworld.openCamera(facing);
    }

    @Override
    public void onPause() {
        super.onPause();
        ncnnyoloworld.closeCamera();
    }
}

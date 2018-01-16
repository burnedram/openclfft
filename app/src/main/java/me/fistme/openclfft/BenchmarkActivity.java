package me.fistme.openclfft;

import android.content.SharedPreferences;
import android.os.Handler;
import android.os.HandlerThread;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.jogamp.opencl.CLDevice;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class BenchmarkActivity extends AppCompatActivity {

    private static final String TAG = "BenchmarkActivity";

    private TextView log;
    private ProgressBar progress;
    private Button benchmark;
    private String[] deviceIds;
    private Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_benchmark);

        boolean autoRun = getIntent().hasExtra("devices");
        deviceIds = getIntent().getStringArrayExtra("devices");

        log = (TextView) findViewById(R.id.log);
        progress = (ProgressBar) findViewById(R.id.progress);
        benchmark = (Button) findViewById(R.id.benchmark);

        HandlerThread t = new HandlerThread("OpenCLFastFourierBenchmark");
        t.start();
        handler = new Handler(t.getLooper());

        if (autoRun)
            benchmark(null);
    }

    @Override
    protected void onDestroy() {
        OpenCLFastFourier.cancelBenchmark();
        super.onDestroy();
    }

    public void benchmark(View v) {
        if (OpenCLFastFourier.isBenchmarkInProgress()) {
            if (v != null) {
                // Cancel pressed
                OpenCLFastFourier.cancelBenchmark();
            }
            return;
        }
        benchmark.setText("Cancel");

        ArrayList<CLDevice> devices;
        if (deviceIds == null)
            devices = null;
        else {
            devices = new ArrayList<>(deviceIds.length);
            for (CLDevice dev : OpenCLFastFourier.getAvailableDevices())
                for (String id : deviceIds)
                    if (OpenCLFastFourier.getId(dev).equals(id)) {
                        devices.add(dev);
                        break;
                    }
        }

        log.setText("==== Benchmark log ====");
        handler.post(() -> {
            OpenCLFastFourier.BenchmarkResults results = OpenCLFastFourier.benchmark(new OpenCLFastFourier.BenchmarkListener() {
                @Override
                public void info(String msg) {
                                           runOnUiThread(() -> log.append("\n" + msg));
                                                                                       }

                @Override
                public void numberOfSteps(int n) {
                                               runOnUiThread(() -> progress.setMax(n));
                                                                                       }

                @Override
                public void step(int n) {
                                      runOnUiThread(() -> progress.setProgress(n));
                                                                                   }
            }, devices);

            runOnUiThread(() -> benchmark.setText("Benchmark"));
            if (results != null && results.bestDevice != null && results.bestDevice.bestConfig != null) {
                OpenCLFastFourier.Config bestConfig = results.bestDevice.bestConfig.config;

                SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
                prefs.edit()
                        .putString("opencl_device", OpenCLFastFourier.getId(bestConfig.device))
                        .putString("opencl_precision", bestConfig.precision.name())
                        .putString("opencl_fftsize", bestConfig.fftSize.name())
                        .apply();

                Log.d(TAG, "OpenCL config from benchmark" +
                        "\n\tDevice: " + bestConfig.device.getName() +
                        "\n\tPrecision: " + bestConfig.precision.name() +
                        "\n\tFFT size: " + bestConfig.fftSize.name());

                OpenCLFastFourier.setConfig(bestConfig);
            }
        });
    }

}

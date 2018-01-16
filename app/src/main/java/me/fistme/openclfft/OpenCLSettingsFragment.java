package me.fistme.openclfft;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceCategory;
import android.preference.PreferenceFragment;
import android.preference.PreferenceGroup;
import android.preference.PreferenceManager;
import android.preference.SwitchPreference;
import android.util.Base64;
import android.util.Log;
import android.util.Xml;

import com.jogamp.opencl.CLDevice;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created by Rafael on 2017-07-12.
 */

public class OpenCLSettingsFragment extends PreferenceFragment implements SharedPreferences.OnSharedPreferenceChangeListener {

    private static final String TAG = "OpenCLSettingsFragment";

    private PreferenceCategory devCat;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.opencl_settings);

        devCat = (PreferenceCategory) findPreference("opencl_devices");
        devCat.removePreference(findPreference("opencl_devices_placeholder"));

        Context ctx = getActivity();
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(ctx);
        prefs.registerOnSharedPreferenceChangeListener(this);

        findPreference("benchmark")
                .setOnPreferenceClickListener(p -> {
                    String[] devices;
                    if (!prefs.getBoolean("opencl_auto_device", false))
                        devices = new String[] {prefs.getString("opencl_device", null)};
                    else {
                        ArrayList<CLDevice> devs = OpenCLFastFourier.getAvailableDevices();
                        devices = new String[devs.size()];
                        for (int i = 0; i < devices.length; i++)
                            devices[i] = OpenCLFastFourier.getId(devs.get(i));
                    }
                    startActivity(new Intent(ctx, BenchmarkActivity.class)
                        .putExtra("devices", devices));
                    return true;
                });

        String currentDevice = prefs.getString("opencl_device", null);
        ArrayList<CLDevice> devices = OpenCLFastFourier.getAvailableDevices();
        for (CLDevice dev : devices) {
            SwitchPreference pref = new SwitchPreference(ctx);
            pref.setTitle(dev.getName());
            pref.setSummary(dev.getPlatform().getName());
            String id = OpenCLFastFourier.getId(dev);
            pref.setKey("opencl_device_" + id);
            pref.setChecked(id.equals(currentDevice));
            devCat.addPreference(pref);
        }
        {
            SwitchPreference pref = new SwitchPreference(ctx);
            pref.setTitle("Test");
            pref.setSummary("Test");
            pref.setKey("opencl_device_test");
            devCat.addPreference(pref);
        }
    }

    @Override
    public void onDestroy() {
        PreferenceManager.getDefaultSharedPreferences(getActivity())
                .unregisterOnSharedPreferenceChangeListener(this);

        super.onDestroy();
    }

    public static void setOpenCLDeviceFrom(SharedPreferences prefs) {
        String devId = prefs.getString("opencl_device", null);
        String precisionName = prefs.getString("opencl_precision", null);
        String fftSizeName = prefs.getString("opencl_fftsize", null);
        if (devId == null || precisionName == null || fftSizeName == null)
            return;

        CLDevice device = OpenCLFastFourier.getAvailableDevices().stream()
                .filter(dev -> OpenCLFastFourier.getId(dev).equals(devId))
                .findAny().orElse(null);
        OpenCLFastFourier.PRECISION precision = OpenCLFastFourier.PRECISION.valueOf(precisionName);
        OpenCLFastFourier.FFT_SIZE fftSize = OpenCLFastFourier.FFT_SIZE.valueOf(fftSizeName);

        Log.d(TAG, "OpenCL config from preferences" +
                "\n\tDevice: " + device.getName() +
                "\n\tPrecesion: " + precisionName +
                "\n\tFFT size: " + fftSizeName);

        OpenCLFastFourier.Config config = new OpenCLFastFourier.Config(device, precision, fftSize);
        OpenCLFastFourier.setConfig(config);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        if (key.startsWith("opencl_device_")) {
            if (sharedPreferences.getBoolean(key, false)) { // if checked...
                for (int i = 0; i < devCat.getPreferenceCount(); i++) {
                    SwitchPreference pref = (SwitchPreference) devCat.getPreference(i);
                    if (!pref.getKey().equals(key))
                        pref.setChecked(false); // ...uncheck all others...
                }
                String id = key.substring("opencl_device_".length());
                sharedPreferences.edit()
                        // save selected device
                        .putString("opencl_device", id)
                        // but it is not benchmarked
                        .remove("opencl_precision")
                        .remove("opencl_fftsize")
                        .apply();
                // unset current selected OpenCL device as well
                OpenCLFastFourier.setConfig(null);
                Log.d(TAG, "OpenCL config cleared");
            } else if (!sharedPreferences.getBoolean("opencl_auto_device", false)) {
                // if manual device selection, make sure at least one is selected all the time
                boolean anyChecked = false;
                for (int i = 0; i < devCat.getPreferenceCount(); i++) {
                    SwitchPreference pref = (SwitchPreference) devCat.getPreference(i);
                    if (pref.isChecked()) {
                        anyChecked = true;
                        break;
                    }
                }
                if (!anyChecked) { // recheck
                    for (int i = 0; i < devCat.getPreferenceCount(); i++) {
                        SwitchPreference pref = (SwitchPreference) devCat.getPreference(i);
                        if (pref.getKey().equals(key)) {
                            pref.setChecked(true);
                            break;
                        }
                    }
                }
            }
        } else {
            switch (key) {
                case "opencl_auto_device": {
                    // Unset current OpenCL device, and related saved prefs
                    sharedPreferences.edit()
                            .remove("opencl_device")
                            .remove("opencl_precision")
                            .remove("opencl_fftsize")
                            .apply();
                    OpenCLFastFourier.setConfig(null);
                    Log.d(TAG, "OpenCL config cleared");

                    if (!sharedPreferences.getBoolean(key, false)) {
                        // Manual selection
                        String id = sharedPreferences.getString("opencl_device", null);
                        if (id != null) {
                            // Check the last checked
                            for (int i = 0; i < devCat.getPreferenceCount(); i++) {
                                SwitchPreference pref = (SwitchPreference) devCat.getPreference(i);
                                pref.setChecked(pref.getKey().equals("opencl_device_" + id));
                            }
                        } else {
                            // Check anything
                            ((SwitchPreference) devCat.getPreference(0)).setChecked(true);
                        }
                    } else {
                        // Automatic, select none
                        for (int i = 0; i < devCat.getPreferenceCount(); i++) {
                            SwitchPreference pref = (SwitchPreference) devCat.getPreference(i);
                            pref.setChecked(false);
                        }
                    }
                    break;
                }
            }
        }
    }
}

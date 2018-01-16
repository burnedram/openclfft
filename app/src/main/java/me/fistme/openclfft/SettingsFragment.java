package me.fistme.openclfft;

import android.os.Bundle;
import android.preference.PreferenceFragment;

/**
 * Created by Rafael on 2017-07-12.
 */

public class SettingsFragment extends PreferenceFragment {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);

        findPreference("opencl").setOnPreferenceClickListener(p -> {
            getFragmentManager()
                    .beginTransaction()
                    .replace(android.R.id.content, new OpenCLSettingsFragment())
                    .addToBackStack(null)
                    .commit();
            return true;
        });
    }

}

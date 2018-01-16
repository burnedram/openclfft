package me.fistme.openclfft;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AutomaticGainControl;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Toast;

import com.jogamp.common.nio.Buffers;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Rafael on 2017-07-12.
 */

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("openclfft");
    }

    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try (InputStream programSource = getResources().openRawResource(R.raw.fft)) {
            OpenCLFastFourier.setProgram(programSource);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Unable to read OpenCL source", Toast.LENGTH_LONG).show();
        }

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        OpenCLSettingsFragment.setOpenCLDeviceFrom(prefs);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 0);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            // The user may be retarded
            Toast.makeText(this, "Are you retarded?", Toast.LENGTH_LONG).show();
            Handler handler = new Handler();
            handler.postDelayed(() -> ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 0), 2000);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.settings:
                startActivity(new Intent(this, SettingsActivity.class));
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private int pIdx = 0, fIdx = 0;
    private OpenCLFastFourier.Config clConfig = new OpenCLFastFourier.Config(
            OpenCLFastFourier.getAvailableDevices().get(0),
            OpenCLFastFourier.PRECISION.values()[0],
            OpenCLFastFourier.FFT_SIZE.values()[0]);

    public void cfg(View v) {
        fIdx = (fIdx+1) % OpenCLFastFourier.FFT_SIZE.values().length;
        if (fIdx == 0)
            pIdx = (pIdx+1) % OpenCLFastFourier.PRECISION.values().length;
        OpenCLFastFourier.PRECISION p = OpenCLFastFourier.PRECISION.values()[pIdx];
        OpenCLFastFourier.FFT_SIZE f = OpenCLFastFourier.FFT_SIZE.values()[fIdx];
        clConfig = new OpenCLFastFourier.Config(
                OpenCLFastFourier.getAvailableDevices().get(0),
                p, f);
    }

    private int readInterval;

    public void tjenna(View v) {
        int src;
        String srcName;
        AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.N &&
                audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) != null) {
            src = MediaRecorder.AudioSource.UNPROCESSED;
            srcName = "UNPROCESSED";
        } else {
            src = MediaRecorder.AudioSource.VOICE_RECOGNITION;
            srcName = "VOICE_RECOGNITION";
        }
        int sampleRate = 44100;
        int config = AudioFormat.CHANNEL_IN_MONO;
        int format = AudioFormat.ENCODING_PCM_16BIT;
        Log.d(TAG, "   Audio source: " + src + " (" + srcName + ")");
        Log.d(TAG, "    Sample rate: " + sampleRate + "hz");
        Log.d(TAG, " Channel config: " + config + " (CHANNEL_IN_MONO)");
        Log.d(TAG, "Encoding format: " + format + " (ENCODING_PCM_16BIT)");

        int minBytes = AudioRecord.getMinBufferSize(sampleRate, config, format);
        int frameSize = 1 * 2; // mono * 16bit
        int maxSamples = sampleRate * 4; // 4 seconds
        int bufSize = Math.max(minBytes, frameSize * maxSamples);
        int bufSizeInSamples = bufSize / frameSize;
        int minReadInterval = sampleRate / 30; // 30 reads per second
        readInterval = minReadInterval;
        ByteBuffer audioBuffer = Buffers.newDirectByteBuffer(bufSize);
        Log.d(TAG, "Minimum audio buffer size: " + minBytes + "b");
        Log.d(TAG, "         Audio frame size: " + frameSize + "b");
        Log.d(TAG, "        Audio buffer size: " + bufSize + "b, " + bufSizeInSamples + " samples, ~" +
                String.format(Locale.US, "%.4f", bufSize / (float) (frameSize * sampleRate)) + "s");
        Log.d(TAG, "         Min read interval: " + readInterval + " samples, ~" +
                String.format(Locale.US, "%.4f", readInterval / (float) sampleRate) + "s");

        AudioRecord audio = new AudioRecord(src, sampleRate, config, format, bufSize);
        if (AutomaticGainControl.isAvailable()) {
            AutomaticGainControl agc = AutomaticGainControl.create(audio.getAudioSessionId());
            agc.setEnabled(false);
        }
        AtomicInteger samplesToRead = new AtomicInteger(0);

        HandlerThread audioThread = new HandlerThread("audioThread");
        audioThread.start();
        Handler audioHandler = new Handler(audioThread.getLooper());

        OpenCLFastFourier<ShortBuffer> fft = new OpenCLFastFourier<>(ShortBuffer.class);

        HandlerThread fftThread = new HandlerThread("fftThread");
        fftThread.start();
        Handler fftHandler = new Handler(fftThread.getLooper(), msg -> {
            int bytesToRead = samplesToRead.getAndSet(0) * frameSize;
            if (bytesToRead == 0)
                return true;

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                audio.read(audioBuffer, bytesToRead, AudioRecord.READ_BLOCKING);
            else {
                while (audioBuffer.position() < bytesToRead) {
                    int read = audio.read(audioBuffer, bytesToRead - audioBuffer.position());
                    audioBuffer.position(audioBuffer.position() + read);
                }
                audioBuffer.rewind();
            }

            FloatBuffer fs = fft.apply(audioBuffer.asShortBuffer(), bytesToRead/frameSize);
            return true;
        });

        audio.setRecordPositionUpdateListener(new AudioRecord.OnRecordPositionUpdateListener() {
            @Override
            public void onMarkerReached(AudioRecord recorder) {
            }

            @Override
            public void onPeriodicNotification(AudioRecord recorder) {
                int oldToRead = samplesToRead.getAndAdd(readInterval);
                fftHandler.sendEmptyMessage(0);

                int oldReadInterval = readInterval;
                if (oldToRead == 0) {
                    // Linearly reduce buffer size (reduce latency, increase FFT runtime)
                    readInterval = Math.max(minReadInterval, readInterval - readInterval / 8);
                } else if (oldToRead > readInterval * 2) {
                    // Quickly increase buffer size (increase latency, reduce FFT runtime)
                    // If oldToRead == readInterval*2, k will be 1. Thus, k > 1.
                    double k = Math.log(oldToRead / (double) readInterval) / Math.log(2);
                    readInterval *= k;
                    if (readInterval > bufSizeInSamples) {
                        Log.w(TAG, "FFT might be to slow, increase audio buffer size");
                        readInterval = bufSizeInSamples;
                    }
                }
                if (oldReadInterval != readInterval)
                    audio.setPositionNotificationPeriod(readInterval);

                /*
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < readInterval/1000; i++)
                    sb.append('=');
                Log.d(TAG, "readInterval: " + sb.toString());
                */
            }
        }, audioHandler);
        audio.setPositionNotificationPeriod(readInterval);
        audio.startRecording();
    }

    public static float toFloat( int hbits )
    {
        int mant = hbits & 0x03ff;            // 10 bits mantissa
        int exp =  hbits & 0x7c00;            // 5 bits exponent
        if( exp == 0x7c00 )                   // NaN/Inf
            exp = 0x3fc00;                    // -> NaN/Inf
        else if( exp != 0 )                   // normalized value
        {
            exp += 0x1c000;                   // exp - 15 + 127
            if( mant == 0 && exp > 0x1c400 )  // smooth transition
                return Float.intBitsToFloat( ( hbits & 0x8000 ) << 16
                        | exp << 13 | 0x3ff );
        }
        else if( mant != 0 )                  // && exp==0 -> subnormal
        {
            exp = 0x1c400;                    // make it normal
            do {
                mant <<= 1;                   // mantissa * 2
                exp -= 0x400;                 // decrease exp by 1
            } while( ( mant & 0x400 ) == 0 ); // while not normal
            mant &= 0x3ff;                    // discard subnormal bit
        }                                     // else +/-0 -> +/-0
        return Float.intBitsToFloat(          // combine all parts
                ( hbits & 0x8000 ) << 16          // sign  << ( 31 - 15 )
                        | ( exp | mant ) << 13 );         // value << ( 23 - 10 )
    }

}

package me.fistme.openclfft;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLEvent;
import com.jogamp.opencl.CLEventList;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLResource;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created by Rafael on 2017-07-09.
 */

public class OpenCLFastFourier<TIn extends Buffer> implements AutoCloseable {

    public static String getId(CLDevice dev) {
        return dev.getPlatform() + "|" + dev.getName();
    }

    public static ArrayList<CLDevice> getAvailableDevices() {
        CLPlatform[] platforms = CLPlatform.listCLPlatforms();
        ArrayList<CLDevice> devices = new ArrayList<>();
        for (CLPlatform p : platforms)
            for (CLDevice d : p.listCLDevices())
                if (d.isAvailable())
                    devices.add(d);
        return devices;
    }

    // some time in the future, fix support for half and double
    // will need to add a step in opencl that converts from half -> float
    public enum PRECISION {
        //HALF(16, "cl_khr_fp16"),
        FLOAT(32, null);
        //DOUBLE(64, "cl_khr_fp64");

        public final int bits;
        private final String extension;
        PRECISION(int bits, String extension) {
            this.bits = bits;
            this.extension = extension;
        }

        public boolean isSupported(CLDevice dev) {
            return extension == null || dev.isExtensionAvailable(extension);
        }
    }

    /**
     * Should reflect the available kernels
     */
    public enum FFT_SIZE {
        TWO(2),
        FOUR(4),
        EIGHT(8),
        SIXTEEN(16),
        THIRTYTWO(32);

        public final int size;
        FFT_SIZE(int size) {
            this.size = size;
        }
    }

    public interface BenchmarkListener {
        void info(String msg);
        void numberOfSteps(int n);
        void step(int n);
    }

    public interface ProfilingListener {
        void onProfilingDone(Profiling p);
    }

    public static class Profiling {
        public final CLEvent writeInput, packAndTwiddle, fft, readOutput;
        public final List<CLEvent> strides;

        private Profiling(CLEventList list) {
            writeInput = list.getEvent(0);
            packAndTwiddle = list.getEvent(1);
            fft = list.getEvent(2);
            ArrayList<CLEvent> strides = new ArrayList<>(list.size() - 4);
            for (int i = 3; i < list.size(); i++)
                strides.add(list.getEvent(i));
            this.strides = Collections.unmodifiableList(strides);
            readOutput = list.getEvent(list.size() - 1);
        }

        public long totalTime() {
            return readOutput.getProfilingInfo(CLEvent.ProfilingCommand.END) -
                    writeInput.getProfilingInfo(CLEvent.ProfilingCommand.SUBMIT);
        }

        public long timeOnDevice() {
            CLEvent lastEvent = strides.isEmpty() ? fft : strides.get(strides.size()-1);
            return lastEvent.getProfilingInfo(CLEvent.ProfilingCommand.END) -
                    packAndTwiddle.getProfilingInfo(CLEvent.ProfilingCommand.START);
        }
    }

    private static AtomicBoolean
            benchmarkInProgress = new AtomicBoolean(false),
            cancelBenchmark = new AtomicBoolean(false);

    public static class BenchmarkResults {
        public final DeviceResults bestDevice;
        public final Map<CLDevice, DeviceResults> results;

        public BenchmarkResults(DeviceResults bestDevice, Map<CLDevice, DeviceResults> results) {
            this.bestDevice = bestDevice;
            this.results = results;
        }
    }

    public static class DeviceResults {
        public final CLDevice device;
        public final ConfigResults bestConfig;
        public final Map<PRECISION, ConfigResults> bestByPrecision;
        public final Map<PRECISION, Map<FFT_SIZE, ConfigResults>> results;

        public DeviceResults(CLDevice device, ConfigResults bestConfig,
                             Map<PRECISION, ConfigResults> bestByPrecision,
                             Map<PRECISION, Map<FFT_SIZE, ConfigResults>> results) {
            this.device = device;
            this.bestConfig = bestConfig;
            this.bestByPrecision = bestByPrecision;
            this.results = results;
        }
    }

    public static class ConfigResults {
        public final Config config;
        public final double fftPerSecond;

        public ConfigResults(Config config, double fftPerSecond) {
            this.config = config;
            this.fftPerSecond = fftPerSecond;
        }
    }

    public static class Config {
        public final CLDevice device;
        public final PRECISION precision;
        public final FFT_SIZE fftSize;

        public Config(CLDevice device, PRECISION precision, FFT_SIZE fftSize) {
            this.device = device;
            this.precision = precision;
            this.fftSize = fftSize;
        }
    }

    public static boolean isBenchmarkInProgress() {
        return benchmarkInProgress.get();
    }

    public static void cancelBenchmark() {
        if (benchmarkInProgress.get())
            cancelBenchmark.set(true);
    }

    public static void benchmark(BenchmarkListener l) {
        benchmark(l, (List<CLDevice>) null);
    }

    public static void benchmark(BenchmarkListener l, CLDevice... devices) {
        benchmark(l, Arrays.asList(devices));
    }

    public static BenchmarkResults benchmark(BenchmarkListener l, List<CLDevice> devices) {
        if (benchmarkInProgress.getAndSet(true))
            throw new IllegalStateException("Benchmark already in progress");
        try {
            if (devices == null)
                devices = getAvailableDevices();
            l.info("   Devices: " + devices.size());

            PRECISION[] precisions = PRECISION.values();
            l.info("Precisions: " + precisions.length);
            FFT_SIZE[] fftSizes = FFT_SIZE.values();
            l.info(" FFT sizes: " + fftSizes.length);
            int nRuns = 256;
            int nSteps = devices.size() * precisions.length * fftSizes.length * nRuns;
            l.numberOfSteps(nSteps);

            int inputSamples = 8 * 1024;
            ShortBuffer input = Buffers.newDirectShortBuffer(inputSamples);
            Random rng = new Random();
            while (input.hasRemaining())
                input.put((short) rng.nextInt());
            input.rewind();

            DeviceResults bestDevice = null;
            HashMap<CLDevice, DeviceResults> results = new HashMap<>();
            for (int devIdx = 0; devIdx < devices.size(); devIdx++) {
                CLDevice dev = devices.get(devIdx);
                CLContext ctx = CLContext.create(dev);
                dev = ctx.getDevices()[0];
                CLProgram program = ctx.createProgram(PROGRAM_SOURCE);
                try {
                    l.info("==== " + dev.getName() + " ====");
                    ConfigResults bestConfig = null;
                    HashMap<PRECISION, ConfigResults> bestByPrecision = new HashMap<>();
                    HashMap<PRECISION, Map<FFT_SIZE, ConfigResults>> configByFftSizeByPrecision = new HashMap<>();

                    for (int precisionIdx = 0; precisionIdx < precisions.length; precisionIdx++) {
                        PRECISION precision = precisions[precisionIdx];
                        if (!precision.isSupported(dev)) {
                            l.info("    " + precision.name().toLowerCase() + " not supported");
                            continue;
                        }
                        ConfigResults bestPrecision = null;
                        HashMap<FFT_SIZE, ConfigResults> configByFftSize = new HashMap<>();

                        for (int fftSizeIdx = 0; fftSizeIdx < fftSizes.length; fftSizeIdx++) {
                            if (cancelBenchmark.getAndSet(false)) {
                                l.info("====Benchmark canceled====");
                                return null;
                            }

                            FFT_SIZE fftSize = fftSizes[fftSizeIdx];
                            try (OpenCLFastFourier<ShortBuffer> fft = new OpenCLFastFourier<>(
                                    ShortBuffer.class,
                                    ctx, dev, program,
                                    precision, fftSize)) {
                                ArrayList<Long> nanos = new ArrayList<>();
                                for (int run = 0; run < nRuns; run++) {
                                    l.step(devIdx * precisions.length * fftSizes.length * nRuns +
                                            precisionIdx * fftSizes.length * nRuns +
                                            fftSizeIdx * nRuns +
                                            run);
                                    fft.setProfilingListener(p -> nanos.add(p.timeOnDevice()));
                                    fft.apply(input, input.capacity());
                                }

                                double mean = 0;
                                for (Long ns : nanos)
                                    mean += ns;
                                mean /= nRuns;
                                //double std = 0;
                                //for (Long ns : nanos)
                                    //std += (ns - mean) * (ns - mean);
                                //std /= nRuns - 1;
                                //std = Math.sqrt(std);
                                double fftPerSecond = 1E9/mean;
                                l.info("    " + precision.name().toLowerCase() + "/" + fftSize.size + ": " +
                                        String.format(Locale.US, "%.2f", fftPerSecond) + " fft/s");

                                Config config = new Config(dev, precision, fftSize);
                                ConfigResults configResults = new ConfigResults(config, fftPerSecond);
                                if (bestPrecision == null || configResults.fftPerSecond > bestPrecision.fftPerSecond)
                                    bestPrecision = configResults;
                                configByFftSize.put(fftSize, configResults);
                            }
                        }

                        if (bestConfig == null || bestPrecision.fftPerSecond > bestConfig.fftPerSecond)
                            bestConfig = bestPrecision;
                        bestByPrecision.put(precision, bestPrecision);
                        configByFftSizeByPrecision.put(precision, Collections.unmodifiableMap(configByFftSize));
                    }

                    DeviceResults deviceResults = new DeviceResults(dev, bestConfig,
                            Collections.unmodifiableMap(bestByPrecision),
                            Collections.unmodifiableMap(configByFftSizeByPrecision));
                    if (bestDevice == null || bestConfig.fftPerSecond > bestDevice.bestConfig.fftPerSecond)
                        bestDevice = deviceResults;
                    results.put(dev, deviceResults);
                } finally {
                    program.release();
                    ctx.release();
                }
            }
            l.step(nSteps);

            BenchmarkResults benchmarkResults = new BenchmarkResults(bestDevice,
                    Collections.unmodifiableMap(results));
            return benchmarkResults;
        } finally {
            cancelBenchmark.set(false);
            benchmarkInProgress.set(false);
        }
    }

    private static String PROGRAM_SOURCE;
    private static Config SELECTED_CONFIG;
    private static CLContext SELECTED_CONTEXT;
    private static CLDevice SELECTED_DEVICE;
    private static CLProgram SELECTED_PROGRAM;
    private static PRECISION SELECTED_PRECISION;
    private static FFT_SIZE SELECTED_FFT_SIZE;

    public static void setProgram(InputStream in) throws IOException {
        char[] b = new char[1024];
        InputStreamReader isr = new InputStreamReader(in, "utf-8");
        StringBuilder sb = new StringBuilder();
        int read;
        while ((read = isr.read(b)) != -1)
            sb.append(b, 0, read);
        setProgram(sb.toString());
    }

    public static void setProgram(String programSource) {
        PROGRAM_SOURCE = programSource;
        if (SELECTED_PROGRAM != null)
            SELECTED_PROGRAM.release();
        SELECTED_PROGRAM = null;
        if (SELECTED_CONTEXT != null)
            SELECTED_PROGRAM = SELECTED_CONTEXT.createProgram(PROGRAM_SOURCE);
    }

    public static void setConfig(Config cfg) {
        SELECTED_FFT_SIZE = null;
        SELECTED_PRECISION = null;
        if (SELECTED_PROGRAM != null)
            SELECTED_PROGRAM.release();
        SELECTED_PROGRAM = null;
        SELECTED_DEVICE = null;
        if (SELECTED_CONTEXT != null)
            SELECTED_CONTEXT.release();
        SELECTED_CONTEXT = null;
        SELECTED_CONFIG = null;

        if (cfg == null)
            return;

        SELECTED_CONFIG = cfg;
        SELECTED_CONTEXT = CLContext.create(cfg.device);
        SELECTED_DEVICE = SELECTED_CONTEXT.getDevices()[0];
        if (PROGRAM_SOURCE != null)
            SELECTED_PROGRAM = SELECTED_CONTEXT.createProgram(PROGRAM_SOURCE);
        SELECTED_PRECISION = cfg.precision;
        SELECTED_FFT_SIZE = cfg.fftSize;
    }

    public static Config getConfig() {
        return SELECTED_CONFIG;
    }

    private static CLProgram build(CLProgram program, Class<? extends Buffer> inType, PRECISION precision) {
        String inTypeStr;
        if (ByteBuffer.class.isAssignableFrom(inType))
            inTypeStr = "char";
        else if (ShortBuffer.class.isAssignableFrom(inType))
            inTypeStr = "short";
        else if (FloatBuffer.class.isAssignableFrom(inType))
            inTypeStr = "float";
        else
            throw new RuntimeException("Input of type " + inType.getSimpleName() + " is not supported");
        return program.build(//CLProgram.CompilerOptions.FAST_RELAXED_MATH,
                "-D IN_TYPE=" + inTypeStr,
                "-D PRECISION=" + precision.bits);
    }

    private final CLContext ctx;
    private final CLDevice dev;
    private final CLKernel kPack, kFft, kStride, kNormalize;

    private final PRECISION precision;
    private final FFT_SIZE fftSize;

    private ProfilingListener profilingListener;

    public OpenCLFastFourier(Class<? extends TIn> inType) {
        this(inType, SELECTED_CONTEXT, SELECTED_DEVICE, SELECTED_PROGRAM, SELECTED_PRECISION, SELECTED_FFT_SIZE);
    }

    private OpenCLFastFourier(Class<? extends TIn> inType,
                              CLContext ctx, CLDevice dev, CLProgram program,
                              PRECISION precision, FFT_SIZE fftSize) {
        this.ctx = ctx;
        this.dev = dev;
        this.precision = precision;
        this.fftSize = fftSize;

        CLProgram prog = build(program, inType, precision);
        // CLProgram.createCLKernel(String) has bad JNI interface
        Map<String, CLKernel> kernels = prog.createCLKernels();
        kPack = kernels.remove("pack_and_twiddle");
        kFft = kernels.remove("apply_fft" + fftSize.size);
        kStride = kernels.remove("apply_stride_fft" + fftSize.size);
        kNormalize = kernels.remove("normalize_fft" + fftSize.size);
        kernels.values().forEach(CLKernel::release); // Release kernels with other FFT sizes
    }

    public void setProfilingListener(ProfilingListener l) {
        profilingListener = l;
    }

    public FloatBuffer apply(TIn in, int nSamples) {
        int log2N = (int) Math.ceil(Math.log(nSamples)/Math.log(2));
        int N = (int) Math.pow(2, log2N);

        int WSize = (N/2 | (N/2 - 1))          - (fftSize.size/2 - 1);
        //          Twiddles for all fft sizes - "Hardcoded" twiddles in fft
        int nKernels = 1+1+1+1+1         + (int) (Math.log(N/fftSize.size)/Math.log(2)) + 1;
        //      write+pack+fft+normalize +             number of fft strides            + read;

        LinkedList<CLResource> toRelease = new LinkedList<>();
        try {
            CLEventList perf = null;
            if (profilingListener != null) {
                perf = new CLEventList(nKernels);
                toRelease.push(perf);
            }

            CLCommandQueue queue = dev.createCommandQueue(CLCommandQueue.Mode.PROFILING_MODE);
            toRelease.push(queue);

            CLBuffer<TIn> clIn = ctx.createBuffer(in, CLMemory.Mem.READ_ONLY);
            toRelease.push(clIn);
            CLBuffer<FloatBuffer> clPack = ctx.createFloatBuffer(2 * N, CLMemory.Mem.READ_WRITE);
            toRelease.push(clPack);
            CLBuffer<FloatBuffer> clW = ctx.createFloatBuffer(precision.bits/8*2 * WSize, CLMemory.Mem.READ_WRITE);
            toRelease.push(clW);

            queue.putWriteBuffer(clIn, false, perf);

            int max_lwx = Math.min(dev.getMaxWorkGroupSize(), dev.getMaxWorkItemSizes()[0]);

            int pack_gwx = N;
            int pack_lwx = Math.min(pack_gwx, max_lwx);
            if (pack_gwx % pack_lwx != 0)
                pack_gwx += pack_lwx - (pack_gwx % pack_lwx);

            kPack.setArgs(clIn, clPack, clW, nSamples, N, fftSize.size, WSize);
            queue.put1DRangeKernel(kPack, 0, pack_gwx, pack_lwx, perf);

            int fft_gwx = N/fftSize.size;
            int fft_lwx = Math.min(fft_gwx, max_lwx);
            if (fft_gwx % fft_lwx != 0)
                fft_gwx += fft_lwx - (fft_gwx % fft_lwx);

            kFft.setArgs(clPack, clW, N);
            queue.put1DRangeKernel(kFft, 0, fft_gwx, fft_lwx, perf);

            int stride_gwx = N / fftSize.size;
            int stride_lwx = Math.min(stride_gwx, max_lwx);
            if (stride_gwx % stride_lwx != 0)
                stride_gwx += stride_lwx - (stride_gwx % stride_lwx);
            for (int stride = fftSize.size * 2; stride <= N; stride *= 2) {
                if (stride < N) {
                    kStride.setArgs(clPack, clW, N, stride);
                    queue.put1DRangeKernel(kStride, 0, stride_gwx, stride_lwx, perf);
                } else {
                    kStride.setArgs(clPack, clW, N/2, stride);
                    queue.put1DRangeKernel(kStride, 0, stride_gwx/2, Math.min(stride_gwx/2, stride_lwx), perf);
                }
            }

            int normalize_gwx = N/fftSize.size;
            int normalize_lwx = Math.min(normalize_gwx, max_lwx);
            if (normalize_gwx % normalize_lwx != 0)
                normalize_gwx += normalize_lwx - (normalize_gwx % normalize_lwx);

            kNormalize.setArgs(clPack, N);
            queue.put1DRangeKernel(kNormalize, 0, normalize_gwx, normalize_lwx, perf);

            queue.putReadBuffer(clPack, true, perf);

            if (profilingListener != null && perf != null)
                profilingListener.onProfilingDone(new Profiling(perf));

            return clPack.getBuffer();
        } finally {
            // Usage of push means that we release in reverse order
            toRelease.forEach(CLResource::release);
        }
    }

    @Override
    public void close() {
        kPack.release();
        kFft.release();
        kStride.release();
        kNormalize.release();
    }

}

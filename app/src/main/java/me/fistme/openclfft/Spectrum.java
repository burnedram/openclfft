package me.fistme.openclfft;

import android.graphics.PixelFormat;
import android.view.Surface;

import com.jogamp.common.nio.Buffers;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * Created by Rafael on 2017-08-17.
 */

public class Spectrum {

    public final int pixelFormat = PixelFormat.RGB_565;
    private final PixelFormat pixelFormatInfo = new PixelFormat();

    public final ByteBuffer buf;
    public final int w, h;
    private int head = 0;

    public Spectrum(int w, int h) {
        PixelFormat.getPixelFormatInfo(pixelFormat, pixelFormatInfo);
        this.w = w;
        this.h = h;
        buf = Buffers.newDirectByteBuffer(w*h*pixelFormatInfo.bytesPerPixel);
    }

    public void shift(FloatBuffer line) {
        if (line.capacity() != h)
            throw new IllegalArgumentException("invalid line height");

        switch (pixelFormat) {
            case PixelFormat.RGB_565:
                shift_rgb565_float(buf, line, head, w, h);
                break;
            default:
                throw new UnsupportedOperationException("not implemented for format " + pixelFormat);
        }

        head = (head + 1) % w;
    }

    private static native void shift_rgb565_float(ByteBuffer buf, FloatBuffer line, int head, int w, int h);

    public static class ANativeWindowLock {
        public final long pWindow;
        public final int sizeOfPixel, stride;
        public final ByteBuffer buf;

        public ANativeWindowLock(long pWindow, int sizeOfPixel, int stride, ByteBuffer buf) {
            this.pWindow = pWindow;
            this.sizeOfPixel = sizeOfPixel;
            this.stride = stride;
            this.buf = buf;
        }
    }

    private static native ANativeWindowLock lock(Surface surface);
    private static native void unlock(long pWindow);

}

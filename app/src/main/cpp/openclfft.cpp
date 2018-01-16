#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

extern "C"
JNIEXPORT jobject JNICALL
Java_me_fistme_openclfft_Spectrum_lock(
        JNIEnv *env, jclass,
        jobject surface) {
    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);
    ANativeWindow_Buffer buffer;

    if (!ANativeWindow_lock(window, &buffer, NULL)) {
        jint sizeof_pixel;
        switch (buffer.format) {
            case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM:
                sizeof_pixel = 2;
                break;
            default:
                ANativeWindow_unlockAndPost(window);
                ANativeWindow_release(window);
                jclass clazz = env->FindClass("java/lang/UnsupportedOperationException");
                env->ThrowNew(clazz, "not implemented for this surface's format");
                return NULL;
        }
        jlong cap = sizeof_pixel * buffer.stride * buffer.height;
        jobject nioBuf = env->NewDirectByteBuffer(buffer.bits, cap);

        jclass clazz = env->FindClass("me/fistme/openclfft/Spectrum$ANativeWindowLock");
        jmethodID constructor = env->GetMethodID(clazz, "<init>", "(JIILjava/nio/ByteBuffer;)V");
        return env->NewObject(clazz, constructor, (jlong)window, sizeof_pixel, (jint)buffer.stride, nioBuf);
    }
    ANativeWindow_unlockAndPost(window);
    ANativeWindow_release(window);
    return NULL;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_me_fistme_openclfft_Spectrum_unlock(
        JNIEnv *, jclass,
        jlong pWindow) {
    ANativeWindow *window = (ANativeWindow *)pWindow;
    ANativeWindow_unlockAndPost(window);
    ANativeWindow_release(window);
}

extern "C"
JNIEXPORT jobject JNICALL
Java_me_fistme_openclfft_Spectrum_shift_1rgb565_1float(
        JNIEnv *env, jclass,
        jobject buf, jobject line, jint head, jint w, jint h) {
    // RGB565
    uint16_t *pBuf = (uint16_t *)env->GetDirectBufferAddress(buf);
    // range [0, 1]
    float *pLine = (float *)env->GetDirectBufferAddress(line);

    //                black, purple, red, yellow, white
    int nGradients = 5;
    float fGradient[] = {0.00, 0.40, 0.60, 0.80, 1.00};
    float hGradient[] = {0.00, 0.83, 0.00, 0.17, 0.00};
    float sGradient[] = {0.00, 1.00, 1.00, 1.00, 0.00};
    float vGradient[] = {0.00, 0.50, 1.00, 1.00, 1.00};

    pBuf = &pBuf[head*sizeof(uint16_t)];
    for (jint y = 0; y < h; y++) {
        // find in which subgradient val exists.
        float val = pLine[y];
        int left = 0, right = nGradients - 1;
        while (left + 1 != right) {
            int mid = left + (right - left)/2;
            if (fGradient[mid] > val)
                right = mid;
            else
                left = mid;
        }

        // interpolate HSV in that subgradient
        val = (val - fGradient[left]) / (fGradient[right] - fGradient[left]);
        float h = hGradient[left] + (hGradient[right] - hGradient[left])*val;
        float s = sGradient[left] + (sGradient[right] - sGradient[left])*val;
        float v = vGradient[left] + (vGradient[right] - vGradient[left])*val;

        // HSV to RGB565
        uint8_t r, g, b;
        h *= 6;
        uint8_t i = (uint8_t) h; // floor(h)
        float c1 = v * (1 - s);
        float c2 = v * (1 - s * (h - i));
        float c3 = v * (1 - s * (1 - (h - i)));
        switch (i) {
            case 0: case 6:
                r = (uint8_t) (v  * 31);
                g = (uint8_t) (c3 * 63);
                b = (uint8_t) (c1 * 31);
                break;
            case 1:
                r = (uint8_t) (c2 * 31);
                g = (uint8_t) (v  * 63);
                b = (uint8_t) (c1 * 31);
                break;
            case 2:
                r = (uint8_t) (c1 * 31);
                g = (uint8_t) (v  * 63);
                b = (uint8_t) (c3 * 31);
                break;
            case 3:
                r = (uint8_t) (c1 * 31);
                g = (uint8_t) (c2 * 63);
                b = (uint8_t) (v  * 31);
                break;
            case 4:
                r = (uint8_t) (c3 * 31);
                g = (uint8_t) (c1 * 63);
                b = (uint8_t) (v  * 31);
                break;
            case 5: default:
                r = (uint8_t) (v  * 31);
                g = (uint8_t) (c1 * 63);
                b = (uint8_t) (c2 * 31);
                break;
        }

        uint16_t rgb565 = (r << 11) | (g << 5) | b;
        pBuf[y * w * sizeof(uint16_t)] = rgb565;
    }
}

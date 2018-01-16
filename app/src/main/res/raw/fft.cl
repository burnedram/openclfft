#ifndef IN_TYPE
#error IN_TYPE not defined
#endif

#ifndef PRECISION
#error PRECISION not defined
#endif

#if PRECISION == 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define F_TYPE half
#define F2_TYPE half2
#define DIV_TYPE float // This is not a typo

#elif PRECISION == 32
#define F_TYPE float
#define F2_TYPE float2
#define DIV_TYPE float

#elif PRECISION == 64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define F_TYPE double
#define F2_TYPE double2
#define DIV_TYPE double

#else
#error Invalid PRECISION
#endif

// Only works if x = 2^n
#define log2(x) (31 - clz(x))

inline size_t floor_log2(size_t x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return popcount(x >> 1);
}

kernel void pack_and_twiddle(global IN_TYPE *in, global F2_TYPE *pack, global F2_TYPE *W,
        const int nSamples, int N, int fftSize, const int WSize) {
    size_t i = get_global_id(0);
    if (i >= N)
        return;

    const uchar bitRev[] =
    {
      0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
      0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
      0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
      0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
      0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
      0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
      0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
      0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
      0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
      0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
      0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
      0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
      0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
      0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
      0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
      0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
    };
    const size_t log2N = log2(N);
    const size_t iRev = (((bitRev[i       & 0xff] << 24) |
                        (bitRev[(i >> 8)  & 0xff] << 16) |
                        (bitRev[(i >> 16) & 0xff] << 8)  |
                        (bitRev[(i >> 24) & 0xff]))     >>
                        (32 - log2N))                    & // We only want to keep the log2N msb
                        ~(((0x1 << 31) >> (32 - log2N)) << 1); // Fix for arithmetic shift operator

    F2_TYPE h;
    h.x        = i >= nSamples ? 0 : //in[i];
        ((in[i]/(DIV_TYPE)((1 << sizeof(IN_TYPE)*8) - 1)) - (DIV_TYPE)0.5) * 2; // Normalize to [-1, 1]
    h.y        = 0;
    pack[iRev] = h;

    if (i >= WSize)
        return;
    fftSize >>= 1;
    const size_t log2FftSize = log2(fftSize);
    const size_t stage = floor_log2((i >> log2FftSize) + 1);
    const size_t logCorrection = log2(N >> 1) - stage;
    const size_t k = (
                ((i >> log2FftSize) // high bits
                - ((1 << stage) - 1)) // correct for stage
                << logCorrection // move back as so to become high bits again
             ) | (
                (i & (fftSize - 1)) // low bits
                << (logCorrection - log2FftSize) // correct for stage
             );

    F_TYPE theta = -(2*k/(DIV_TYPE)N);
    W[i]         = (F2_TYPE) (cospi(theta), sinpi(theta));
}

#define fft2(v0, v1, t) \
do {              \
    t  = v0 + v1; \
    v1 = v0 - v1; \
    v0 = t;       \
} while(0)

#define fft4(v0, v1, v2, v3, t) \
do {              \
    t  = v0 + v2; \
    v2 = v0 - v2; \
    v0 = t;       \
    t  = v1 + v3; \
    v3 = v1 - v3; \
    v1 = t;       \
} while(0)

#define fft8(v0, v1, v2, v3, v4, v5, v6, v7, t) \
do {              \
    t  = v0 + v4; \
    v4 = v0 - v4; \
    v0 = t;       \
    t  = v1 + v5; \
    v5 = v1 - v5; \
    v1 = t;       \
    t  = v2 + v6; \
    v6 = v2 - v6; \
    v2 = t;       \
    t  = v3 + v7; \
    v7 = v3 - v7; \
    v3 = t;       \
} while(0)

#define fft16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, t) \
do {                \
    t   = v0 + v8;  \
    v8  = v0 - v8;  \
    v0  = t;        \
    t   = v1 + v9;  \
    v9  = v1 - v9;  \
    v1  = t;        \
    t   = v2 + v10; \
    v10 = v2 - v10; \
    v2  = t;        \
    t   = v3 + v11; \
    v11 = v3 - v11; \
    v3  = t;        \
    t   = v4 + v12; \
    v12 = v4 - v12; \
    v4 = t;        \
    t   = v5 + v13; \
    v13 = v5 - v13; \
    v5  = t;        \
    t   = v6 + v14; \
    v14 = v6 - v14; \
    v6 = t;        \
    t   = v7 + v15; \
    v15 = v7 - v15; \
    v7  = t;        \
} while(0)

#define fft32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, \
        v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, t) \
do {                 \
    t   = v0 + v16;  \
    v16 = v0 - v16;  \
    v0  = t;         \
    t   = v1 + v17;  \
    v17 = v1 - v17;  \
    v1  = t;         \
    t   = v2 + v18;  \
    v18 = v2 - v18;  \
    v2  = t;         \
    t   = v3 + v19;  \
    v19 = v3 - v19;  \
    v3  = t;         \
    t   = v4 + v20;  \
    v20 = v4 - v20;  \
    v4 = t;          \
    t   = v5 + v21;  \
    v21 = v5 - v21;  \
    v5  = t;         \
    t   = v6 + v22;  \
    v22 = v6 - v22;  \
    v6 = t;          \
    t   = v7 + v23;  \
    v23 = v7 - v23;  \
    v7  = t;         \
    t   = v8 + v24;  \
    v24 = v8 - v24;  \
    v8  = t;         \
    t   = v9 + v25;  \
    v25 = v9 - v25;  \
    v9  = t;         \
    t   = v10 + v26; \
    v26 = v10 - v26; \
    v10  = t;        \
    t   = v11 + v27; \
    v27 = v11 - v27; \
    v11  = t;        \
    t   = v12 + v28; \
    v28 = v12 - v28; \
    v12  = t;        \
    t   = v13 + v29; \
    v29 = v13 - v29; \
    v13  = t;        \
    t   = v14 + v30; \
    v30 = v14 - v30; \
    v14  = t;        \
    t   = v15 + v31; \
    v31 = v15 - v31; \
    v15  = t;        \
} while(0)

#define cpx_mul(v0, v1, t) \
do {                             \
    t.x = v0.x*v1.x - v0.y*v1.y; \
    t.y = v0.x*v1.y + v0.y*v1.x; \
    v0  = t;                     \
} while(0)

kernel void apply_fft2(global F2_TYPE *pack, global F2_TYPE *W,
        const int N) {
    const size_t i = get_global_id(0)*2;
    if (i >= N)
        return;

    pack = &pack[i];
    F2_TYPE t,
            v0 = pack[0],
            v1 = pack[1];

    fft2(v0, v1, t);

    pack[0] = v0;
    pack[1] = v1;
}

kernel void apply_stride_fft2(global F2_TYPE *pack, global F2_TYPE *W,
        const int N, const int fullStride) {
    size_t i             = get_global_id(0)*1;
    const size_t stride  = fullStride >> 1;
    const size_t group   = i / stride;
    const size_t inGroup = i % stride;
    i                    = group*fullStride + inGroup;
    if (i >= N)
        return;

    pack                       = &pack[i];
    global F2_TYPE *packStride = &pack[stride];
    W                          = &W[((stride - 1) & ~0x00) + inGroup];

    F2_TYPE t,
            v0 = pack[0],
            v1 = packStride[0];

    cpx_mul(v1, W[0], t);

    fft2(v0, v1, t);

    pack[0]       = v0;
    packStride[0] = v1;
}

kernel void normalize_fft2(global F2_TYPE *pack,
        const int N) {
    const size_t i = get_global_id(0)*2;
    if (i >= N)
        return;

    pack = &pack[i];
    pack[0] = pack[0]/N;
    pack[1] = pack[1]/N;
}

kernel void apply_fft4(global F2_TYPE *pack, global F2_TYPE *W,
        const int N) {
    const size_t i = get_global_id(0)*4;
    if (i >= N)
        return;

    pack = &pack[i];
    F2_TYPE t,
            v0 = pack[0],
            v1 = pack[1],
            v2 = pack[2],
            v3 = pack[3];

    fft2(v0, v1, t);
    fft2(v2, v3, t);

    cpx_mul(v2,  W[0], t);
    cpx_mul(v3,  W[1], t);

    fft4(v0, v1, v2, v3, t);

    pack[0] = v0;
    pack[1] = v1;
    pack[2] = v2;
    pack[3] = v3;
}

kernel void apply_stride_fft4(global F2_TYPE *pack, global F2_TYPE *W,
        const int N, const int fullStride) {
    size_t i             = get_global_id(0)*2;
    const size_t stride  = fullStride >> 1;
    const size_t group   = i / stride;
    const size_t inGroup = i % stride;
    i                    = group*fullStride + inGroup;
    if (i >= N)
        return;

    pack                       = &pack[i];
    global F2_TYPE *packStride = &pack[stride];
    W                          = &W[((stride - 1) & ~0x01) + inGroup];

    F2_TYPE t,
            v0 = pack[0],
            v1 = pack[1],
            v2 = packStride[0],
            v3 = packStride[1];

    cpx_mul(v2, W[0], t);
    cpx_mul(v3, W[1], t);

    fft4(v0, v1, v2, v3, t);

    pack[0]       = v0;
    pack[1]       = v1;
    packStride[0] = v2;
    packStride[1] = v3;
}

kernel void normalize_fft4(global F2_TYPE *pack,
        const int N) {
    const size_t i = get_global_id(0)*4;
    if (i >= N)
        return;

    pack = &pack[i];
    pack[0] = pack[0]/N;
    pack[1] = pack[1]/N;
    pack[2] = pack[2]/N;
    pack[3] = pack[3]/N;
}

kernel void apply_fft8(global F2_TYPE *pack, global F2_TYPE *W,
        const int N) {
    const size_t i = get_global_id(0)*8;
    if (i >= N)
        return;

    pack = &pack[i];
    F2_TYPE t,
            v0 = pack[0],
            v1 = pack[1],
            v2 = pack[2],
            v3 = pack[3],
            v4 = pack[4],
            v5 = pack[5],
            v6 = pack[6],
            v7 = pack[7];

    fft2(v0, v1, t);
    fft2(v2, v3, t);
    fft2(v4, v5, t);
    fft2(v6, v7, t);

    cpx_mul(v2,  W[0], t);
    cpx_mul(v6,  W[0], t);
    cpx_mul(v3,  W[2], t);
    cpx_mul(v7,  W[2], t);

    fft4(v0, v1, v2, v3, t);
    fft4(v4, v5, v6, v7, t);

    cpx_mul(v4,  W[0], t);
    cpx_mul(v5,  W[1], t);
    cpx_mul(v6,  W[2], t);
    cpx_mul(v7,  W[3], t);

    fft8(v0, v1, v2, v3, v4, v5, v6, v7, t);

    pack[0] = v0;
    pack[1] = v1;
    pack[2] = v2;
    pack[3] = v3;
    pack[4] = v4;
    pack[5] = v5;
    pack[6] = v6;
    pack[7] = v7;
}

kernel void apply_stride_fft8(global F2_TYPE *pack, global F2_TYPE *W,
        const int N, const int fullStride) {
    size_t i             = get_global_id(0)*4;
    const size_t stride  = fullStride >> 1;
    const size_t group   = i / stride;
    const size_t inGroup = i % stride;
    i                    = group*fullStride + inGroup;
    if (i >= N)
        return;

    pack                       = &pack[i];
    global F2_TYPE *packStride = &pack[stride];
    W                          = &W[((stride - 1) & ~0x03) + inGroup];

    F2_TYPE t,
            v0 = pack[0],
            v1 = pack[1],
            v2 = pack[2],
            v3 = pack[3],
            v4 = packStride[0],
            v5 = packStride[1],
            v6 = packStride[2],
            v7 = packStride[3];

    cpx_mul(v4, W[0], t);
    cpx_mul(v5, W[1], t);
    cpx_mul(v6, W[2], t);
    cpx_mul(v7, W[3], t);

    fft8(v0, v1, v2, v3, v4, v5, v6, v7, t);

    pack[0]       = v0;
    pack[1]       = v1;
    pack[2]       = v2;
    pack[3]       = v3;
    packStride[0] = v4;
    packStride[1] = v5;
    packStride[2] = v6;
    packStride[3] = v7;
}

kernel void normalize_fft8(global F2_TYPE *pack,
        const int N) {
    const size_t i = get_global_id(0)*8;
    if (i >= N)
        return;

    pack = &pack[i];
    pack[0] = pack[0]/N;
    pack[1] = pack[1]/N;
    pack[2] = pack[2]/N;
    pack[3] = pack[3]/N;
    pack[4] = pack[4]/N;
    pack[5] = pack[5]/N;
    pack[6] = pack[6]/N;
    pack[7] = pack[7]/N;
}

kernel void apply_fft16(global F2_TYPE *pack, global F2_TYPE *W,
        const int N) {
    const size_t i = get_global_id(0)*16;
    if (i >= N)
        return;

    pack = &pack[i];
    F2_TYPE t,
            v0  = pack[0],
            v1  = pack[1],
            v2  = pack[2],
            v3  = pack[3],
            v4  = pack[4],
            v5  = pack[5],
            v6  = pack[6],
            v7  = pack[7],
            v8  = pack[8],
            v9  = pack[9],
            v10 = pack[10],
            v11 = pack[11],
            v12 = pack[12],
            v13 = pack[13],
            v14 = pack[14],
            v15 = pack[15];

    fft2(v0, v1, t);
    fft2(v2, v3, t);
    fft2(v4, v5, t);
    fft2(v6, v7, t);
    fft2(v8, v9, t);
    fft2(v10, v11, t);
    fft2(v12, v13, t);
    fft2(v14, v15, t);

    cpx_mul(v2,  W[0], t);
    cpx_mul(v6,  W[0], t);
    cpx_mul(v10, W[0], t);
    cpx_mul(v14, W[0], t);
    cpx_mul(v3,  W[4], t);
    cpx_mul(v7,  W[4], t);
    cpx_mul(v11, W[4], t);
    cpx_mul(v15, W[4], t);

    fft4(v0, v1, v2, v3, t);
    fft4(v4, v5, v6, v7, t);
    fft4(v8, v9, v10, v11, t);
    fft4(v12, v13, v14, v15, t);

    cpx_mul(v4,  W[0], t);
    cpx_mul(v12, W[0], t);
    cpx_mul(v5,  W[2], t);
    cpx_mul(v13, W[2], t);
    cpx_mul(v6,  W[4], t);
    cpx_mul(v14, W[4], t);
    cpx_mul(v7,  W[6], t);
    cpx_mul(v15, W[6], t);

    fft8(v0, v1, v2, v3, v4, v5, v6, v7, t);
    fft8(v8, v9, v10, v11, v12, v13, v14, v15, t);

    cpx_mul(v8,  W[0], t);
    cpx_mul(v9,  W[1], t);
    cpx_mul(v10, W[2], t);
    cpx_mul(v11, W[3], t);
    cpx_mul(v12, W[4], t);
    cpx_mul(v13, W[5], t);
    cpx_mul(v14, W[6], t);
    cpx_mul(v15, W[7], t);

    fft16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, t);

    pack[0]  = v0;
    pack[1]  = v1;
    pack[2]  = v2;
    pack[3]  = v3;
    pack[4]  = v4;
    pack[5]  = v5;
    pack[6]  = v6;
    pack[7]  = v7;
    pack[8]  = v8;
    pack[9]  = v9;
    pack[10] = v10;
    pack[11] = v11;
    pack[12] = v12;
    pack[13] = v13;
    pack[14] = v14;
    pack[15] = v15;
}

kernel void apply_stride_fft16(global F2_TYPE *pack, global F2_TYPE *W,
        const int N, const int fullStride) {
    size_t i             = get_global_id(0)*8;
    const size_t stride  = fullStride >> 1;
    const size_t group   = i / stride;
    const size_t inGroup = i % stride;
    i                    = group*fullStride + inGroup;
    if (i >= N)
        return;

    pack                       = &pack[i];
    global F2_TYPE *packStride = &pack[stride];
    W                          = &W[((stride - 1) & ~0x07) + inGroup];

    F2_TYPE t,
            v0  = pack[0],
            v1  = pack[1],
            v2  = pack[2],
            v3  = pack[3],
            v4  = pack[4],
            v5  = pack[5],
            v6  = pack[6],
            v7  = pack[7],
            v8  = packStride[0],
            v9  = packStride[1],
            v10 = packStride[2],
            v11 = packStride[3],
            v12 = packStride[4],
            v13 = packStride[5],
            v14 = packStride[6],
            v15 = packStride[7];

    cpx_mul(v8,  W[0], t);
    cpx_mul(v9,  W[1], t);
    cpx_mul(v10, W[2], t);
    cpx_mul(v11, W[3], t);
    cpx_mul(v12, W[4], t);
    cpx_mul(v13, W[5], t);
    cpx_mul(v14, W[6], t);
    cpx_mul(v15, W[7], t);

    fft16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, t);

    pack[0]       = v0;
    pack[1]       = v1;
    pack[2]       = v2;
    pack[3]       = v3;
    pack[4]       = v4;
    pack[5]       = v5;
    pack[6]       = v6;
    pack[7]       = v7;
    packStride[0] = v8;
    packStride[1] = v9;
    packStride[2] = v10;
    packStride[3] = v11;
    packStride[4] = v12;
    packStride[5] = v13;
    packStride[6] = v14;
    packStride[7] = v15;
}

kernel void normalize_fft16(global F2_TYPE *pack,
        const int N) {
    const size_t i = get_global_id(0)*16;
    if (i >= N)
        return;

    pack = &pack[i];
    pack[0] = pack[0]/N;
    pack[1] = pack[1]/N;
    pack[2] = pack[2]/N;
    pack[3] = pack[3]/N;
    pack[4] = pack[4]/N;
    pack[5] = pack[5]/N;
    pack[6] = pack[6]/N;
    pack[7] = pack[7]/N;
    pack[8] = pack[8]/N;
    pack[9] = pack[9]/N;
    pack[10] = pack[10]/N;
    pack[11] = pack[11]/N;
    pack[12] = pack[12]/N;
    pack[13] = pack[13]/N;
    pack[14] = pack[14]/N;
    pack[15] = pack[15]/N;
}

kernel void apply_fft32(global F2_TYPE *pack, global F2_TYPE *W,
        const int N) {
    const size_t i = get_global_id(0)*32;
    if (i >= N)
        return;

    pack = &pack[i];
    F2_TYPE t,
            v0  = pack[0],
            v1  = pack[1],
            v2  = pack[2],
            v3  = pack[3],
            v4  = pack[4],
            v5  = pack[5],
            v6  = pack[6],
            v7  = pack[7],
            v8  = pack[8],
            v9  = pack[9],
            v10 = pack[10],
            v11 = pack[11],
            v12 = pack[12],
            v13 = pack[13],
            v14 = pack[14],
            v15 = pack[15],
            v16 = pack[16],
            v17 = pack[17],
            v18 = pack[18],
            v19 = pack[19],
            v20 = pack[20],
            v21 = pack[21],
            v22 = pack[22],
            v23 = pack[23],
            v24 = pack[24],
            v25 = pack[25],
            v26 = pack[26],
            v27 = pack[27],
            v28 = pack[28],
            v29 = pack[29],
            v30 = pack[30],
            v31 = pack[31];

    fft2(v0, v1, t);
    fft2(v2, v3, t);
    fft2(v4, v5, t);
    fft2(v6, v7, t);
    fft2(v8, v9, t);
    fft2(v10, v11, t);
    fft2(v12, v13, t);
    fft2(v14, v15, t);
    fft2(v16, v17, t);
    fft2(v18, v19, t);
    fft2(v20, v21, t);
    fft2(v22, v23, t);
    fft2(v24, v25, t);
    fft2(v26, v27, t);
    fft2(v28, v29, t);
    fft2(v30, v31, t);

    cpx_mul(v2,  W[0], t);
    cpx_mul(v6,  W[0], t);
    cpx_mul(v10, W[0], t);
    cpx_mul(v14, W[0], t);
    cpx_mul(v18, W[0], t);
    cpx_mul(v22, W[0], t);
    cpx_mul(v26, W[0], t);
    cpx_mul(v30, W[0], t);
    cpx_mul(v3,  W[8], t);
    cpx_mul(v7,  W[8], t);
    cpx_mul(v11, W[8], t);
    cpx_mul(v15, W[8], t);
    cpx_mul(v19, W[8], t);
    cpx_mul(v23, W[8], t);
    cpx_mul(v27, W[8], t);
    cpx_mul(v31, W[8], t);

    fft4(v0, v1, v2, v3, t);
    fft4(v4, v5, v6, v7, t);
    fft4(v8, v9, v10, v11, t);
    fft4(v12, v13, v14, v15, t);
    fft4(v16, v17, v18, v19, t);
    fft4(v20, v21, v22, v23, t);
    fft4(v24, v25, v26, v27, t);
    fft4(v28, v29, v30, v31, t);

    cpx_mul(v4,  W[0],  t);
    cpx_mul(v12, W[0],  t);
    cpx_mul(v20, W[0],  t);
    cpx_mul(v28, W[0],  t);
    cpx_mul(v5,  W[4],  t);
    cpx_mul(v13, W[4],  t);
    cpx_mul(v21, W[4],  t);
    cpx_mul(v29, W[4],  t);
    cpx_mul(v6,  W[8],  t);
    cpx_mul(v14, W[8],  t);
    cpx_mul(v22, W[8],  t);
    cpx_mul(v30, W[8],  t);
    cpx_mul(v7,  W[12], t);
    cpx_mul(v15, W[12], t);
    cpx_mul(v23, W[12], t);
    cpx_mul(v31, W[12], t);

    fft8(v0, v1, v2, v3, v4, v5, v6, v7, t);
    fft8(v8, v9, v10, v11, v12, v13, v14, v15, t);
    fft8(v16, v17, v18, v19, v20, v21, v22, v23, t);
    fft8(v24, v25, v26, v27, v28, v29, v30, v31, t);

    cpx_mul(v8,  W[0], t);
    cpx_mul(v24, W[0], t);
    cpx_mul(v9,  W[2], t);
    cpx_mul(v25, W[2], t);
    cpx_mul(v10, W[4], t);
    cpx_mul(v26, W[4], t);
    cpx_mul(v11, W[6], t);
    cpx_mul(v27, W[6], t);
    cpx_mul(v12, W[8], t);
    cpx_mul(v28, W[8], t);
    cpx_mul(v13, W[10], t);
    cpx_mul(v29, W[10], t);
    cpx_mul(v14, W[12], t);
    cpx_mul(v30, W[12], t);
    cpx_mul(v15, W[14], t);
    cpx_mul(v31, W[14], t);

    fft16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, t);
    fft16(v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, t);

    cpx_mul(v16, W[0], t);
    cpx_mul(v17, W[1], t);
    cpx_mul(v18, W[2], t);
    cpx_mul(v19, W[3], t);
    cpx_mul(v20, W[4], t);
    cpx_mul(v21, W[5], t);
    cpx_mul(v22, W[6], t);
    cpx_mul(v23, W[7], t);
    cpx_mul(v24, W[8], t);
    cpx_mul(v25, W[9], t);
    cpx_mul(v26, W[10], t);
    cpx_mul(v27, W[11], t);
    cpx_mul(v28, W[12], t);
    cpx_mul(v29, W[13], t);
    cpx_mul(v30, W[14], t);
    cpx_mul(v31, W[15], t);

    fft32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, t);

    pack[0]  = v0;
    pack[1]  = v1;
    pack[2]  = v2;
    pack[3]  = v3;
    pack[4]  = v4;
    pack[5]  = v5;
    pack[6]  = v6;
    pack[7]  = v7;
    pack[8]  = v8;
    pack[9]  = v9;
    pack[10] = v10;
    pack[11] = v11;
    pack[12] = v12;
    pack[13] = v13;
    pack[14] = v14;
    pack[15] = v15;
    pack[16] = v16;
    pack[17] = v17;
    pack[18] = v18;
    pack[19] = v19;
    pack[20] = v20;
    pack[21] = v21;
    pack[22] = v22;
    pack[23] = v23;
    pack[24] = v24;
    pack[25] = v25;
    pack[26] = v26;
    pack[27] = v27;
    pack[28] = v28;
    pack[29] = v29;
    pack[30] = v30;
    pack[31] = v31;
}

kernel void apply_stride_fft32(global F2_TYPE *pack, global F2_TYPE *W,
        const int N, const int fullStride) {
    size_t i             = get_global_id(0)*16;
    const size_t stride  = fullStride >> 1;
    const size_t group   = i / stride;
    const size_t inGroup = i % stride;
    i                    = group*fullStride + inGroup;
    if (i >= N)
        return;

    pack                       = &pack[i];
    global F2_TYPE *packStride = &pack[stride];
    W                          = &W[((stride - 1) & ~0x0F) + inGroup];

    F2_TYPE t,
            v0  = pack[0],
            v1  = pack[1],
            v2  = pack[2],
            v3  = pack[3],
            v4  = pack[4],
            v5  = pack[5],
            v6  = pack[6],
            v7  = pack[7],
            v8  = pack[8],
            v9  = pack[9],
            v10 = pack[10],
            v11 = pack[11],
            v12 = pack[12],
            v13 = pack[13],
            v14 = pack[14],
            v15 = pack[15],
            v16 = packStride[0],
            v17 = packStride[1],
            v18 = packStride[2],
            v19 = packStride[3],
            v20 = packStride[4],
            v21 = packStride[5],
            v22 = packStride[6],
            v23 = packStride[7],
            v24 = packStride[8],
            v25 = packStride[9],
            v26 = packStride[10],
            v27 = packStride[11],
            v28 = packStride[12],
            v29 = packStride[13],
            v30 = packStride[14],
            v31 = packStride[15];

    cpx_mul(v16, W[0], t);
    cpx_mul(v17, W[1], t);
    cpx_mul(v18, W[2], t);
    cpx_mul(v19, W[3], t);
    cpx_mul(v20, W[4], t);
    cpx_mul(v21, W[5], t);
    cpx_mul(v22, W[6], t);
    cpx_mul(v23, W[7], t);
    cpx_mul(v24, W[8], t);
    cpx_mul(v25, W[9], t);
    cpx_mul(v26, W[10], t);
    cpx_mul(v27, W[11], t);
    cpx_mul(v28, W[12], t);
    cpx_mul(v29, W[13], t);
    cpx_mul(v30, W[14], t);
    cpx_mul(v31, W[15], t);

    fft32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, t);

    pack[0]        = v0;
    pack[1]        = v1;
    pack[2]        = v2;
    pack[3]        = v3;
    pack[4]        = v4;
    pack[5]        = v5;
    pack[6]        = v6;
    pack[7]        = v7;
    pack[8]        = v8;
    pack[9]        = v9;
    pack[10]       = v10;
    pack[11]       = v11;
    pack[12]       = v12;
    pack[13]       = v13;
    pack[14]       = v14;
    pack[15]       = v15;
    packStride[0]  = v16;
    packStride[1]  = v17;
    packStride[2]  = v18;
    packStride[3]  = v19;
    packStride[4]  = v20;
    packStride[5]  = v21;
    packStride[6]  = v22;
    packStride[7]  = v23;
    packStride[8]  = v24;
    packStride[9]  = v25;
    packStride[10] = v26;
    packStride[11] = v27;
    packStride[12] = v28;
    packStride[13] = v29;
    packStride[14] = v30;
    packStride[15] = v31;
}

kernel void normalize_fft32(global F2_TYPE *pack,
        const int N) {
    const size_t i = get_global_id(0)*32;
    if (i >= N)
        return;

    pack = &pack[i];
    pack[0] = pack[0]/N;
    pack[1] = pack[1]/N;
    pack[2] = pack[2]/N;
    pack[3] = pack[3]/N;
    pack[4] = pack[4]/N;
    pack[5] = pack[5]/N;
    pack[6] = pack[6]/N;
    pack[7] = pack[7]/N;
    pack[8] = pack[8]/N;
    pack[9] = pack[9]/N;
    pack[10] = pack[10]/N;
    pack[11] = pack[11]/N;
    pack[12] = pack[12]/N;
    pack[13] = pack[13]/N;
    pack[14] = pack[14]/N;
    pack[15] = pack[15]/N;
    pack[16] = pack[16]/N;
    pack[17] = pack[17]/N;
    pack[18] = pack[18]/N;
    pack[19] = pack[19]/N;
    pack[20] = pack[20]/N;
    pack[21] = pack[21]/N;
    pack[22] = pack[22]/N;
    pack[23] = pack[23]/N;
    pack[24] = pack[24]/N;
    pack[25] = pack[25]/N;
    pack[26] = pack[26]/N;
    pack[27] = pack[27]/N;
    pack[28] = pack[28]/N;
    pack[29] = pack[29]/N;
    pack[30] = pack[30]/N;
    pack[31] = pack[31]/N;
}

#ifdef __aarch64__
#define for_each_reg(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8) f(9) f(10) f(11) \
              f(12) f(13) f(14) f(15) f(16) f(17) f(18) f(19) f(20) f(21) f(22) f(23)
#define for_each_reg2(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8) f(9) f(10) f(11) \
              f(12) f(13) f(14) f(15) f(16) f(17) f(18) f(19) f(20) f(21) f(22) f(23)
#else
#define for_each_reg(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8)
#define for_each_reg2(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8)
#endif

// float
// see google's gemmlowp for faster implementation using assembly
#define float_op(x) if (x < num_reg) \
    v##x = vreinterpretq_u8_f32(vmlaq_f32(vreinterpretq_f32_u8(v##x), in0, *_w++));

#define float_kernel(in, weight, depth) ({ \
    float *_i = (void*) (in); \
    float32x4_t *_w = (void*) (weight); \
    uint _d = (depth); \
    for (int z = 0; z < _d; z++) { \
        float32x4_t in0 = vdupq_n_f32(_i[z]); \
        for_each_reg(float_op) \
    } \
})

// int8
// could accumulate two 16bit results with a single pairwise accumulate instead of 2x addw
#define int8_op(x) if (x < num_reg) ({ \
    int16x8_t t[4]; \
    t[0] = vmull_s8(vget_low_s8(_w[x /8*4]), in0); \
    t[1] = vmull_s8(vget_high_s8(_w[x /8*4]), in0); \
    t[2] = vmull_s8(vget_low_s8(_w[x /8*4 + 1]), in0); \
    t[3] = vmull_s8(vget_high_s8(_w[x /8*4 + 1]), in0); \
    t[0] = vmlal_s8(t[0], vget_low_s8(_w[x /8*4 + 2]), in1); \
    t[1] = vmlal_s8(t[1], vget_high_s8(_w[x /8*4 + 2]), in1); \
    t[2] = vmlal_s8(t[2], vget_low_s8(_w[x /8*4 + 3]), in1); \
    t[3] = vmlal_s8(t[3], vget_high_s8(_w[x /8*4 + 3]), in1); \
    v##x = vreinterpretq_u8_s32(vaddw_s16(vreinterpretq_s32_u8(v##x), (x & 1) ? vget_high_s16(t[x / 2 & 3]) : vget_low_s16(t[x / 2 & 3]))); \
});

#define int8_op2(x) if (x < num_reg) ({ \
    int16x8_t t[2]; \
    t[0] = vmull_s8(vget_low_s8(_w[x / 4]), in0); \
    t[1] = vmull_s8(vget_high_s8(_w[x / 4]), in0); \
    v##x = vreinterpretq_u8_s32(vaddw_s16(vreinterpretq_s32_u8(v##x), (x & 1) ? vget_high_s16(t[x / 2 & 1]) : vget_low_s16(t[x / 2 & 1]))); \
});

#define int8_kernel(in, weight, depth) ({ \
    int8_t *_i = (void*) (in); \
    int8x16_t *_w = (void*) (weight); \
    uint _d = (depth); \
    int z; \
    for (z = 0; z < (_d & ~1); z += 2) { \
        int8x8_t in0 = vdup_n_s8(_i[z]); \
        int8x8_t in1 = vdup_n_s8(_i[z+1]); \
        for_each_reg(int8_op); \
        _w += num_reg / 2; \
    } \
    for (; z < _d; z++) { \
        int8x8_t in0 = vdup_n_s8(_i[z]); \
        for_each_reg(int8_op2); \
        _w += num_reg / 4; \
    } \
})

// XOR
// TODO
#define bin_op(x) if (x < num_reg) ({ \
    uint8x16_t tmp = vcntq_u8(_w[x / 2] ^ in0); \
    v##x = vreinterpretq_u8_u16(vaddw_u8(vreinterpretq_u16_u8(v##x), (x & 1) ? vget_high_u8(tmp) : vget_low_u8(tmp))); \
});

#define bin_kernel(in, weight, depth) ({ \
    uint8_t *_i = (void*) (in); \
    uint8x16_t *_w = (void*) (weight); \
    uint _d = (depth); \
    int z; \
    for (z = 0; z < _d / 8; z++) { \
        uint8x16_t in0 = vdupq_n_u8(_i[z]); \
        for_each_reg(bin_op); \
        _w += num_reg / 2; \
    } \
})

// float-binary
// debinarization is costly
#define float_bin_op(x) if (x < num_reg) ({ \
    float32x4_t sel; \
    uint32x4_t tmp[4]; \
    tmp[0] = vtstq_u32(vdupq_n_u32(_w[x / 4]), mask0); \
    tmp[1] = vtstq_u32(vdupq_n_u32(_w[x / 4]), mask1); \
    tmp[2] = vtstq_u32(vdupq_n_u32(_w[x / 4]), mask2); \
    tmp[3] = vtstq_u32(vdupq_n_u32(_w[x / 4]), mask3); \
    sel = vbslq_f32(tmp[x % 4], in1, in0); \
    v##x = vreinterpretq_u8_f32(vaddq_f32(vreinterpretq_f32_u8(v##x), sel)); \
});

#define float_bin_kernel(in, weight, depth) ({ \
    float *_i = (void*) (in); \
    uint16_t *_w = (void*) (weight); \
    uint32x4_t mask0 = {128, 64, 32, 16}; \
    uint32x4_t mask1 = {8, 4, 2, 1}; \
    uint32x4_t mask2 = {32768, 16384, 8192, 4096}; \
    uint32x4_t mask3 = {2048, 1024, 512, 256}; \
    uint _d = (depth); \
    int z; \
    for (z = 0; z < _d; z++) { \
        float32x4_t in0 = vdupq_n_f32(_i[z]); \
        float32x4_t in1 = vdupq_n_f32(-_i[z]); \
        for_each_reg(float_bin_op); \
        _w += num_reg / 4; \
    } \
})


// uint8-binary
// 1. debinarize into 8-bit masks with cmtst
// 2. multiply by 0/1 using AND instruction (the result must be corrected later)
// 3. accumulate pairwise into 16-bit accumulators
// -accumulate up to 256 results into 16-bit registers then add to 32-bit register
// -in some cases only 16-bit accumulators would be needed..
// -clang likes to replace cmtst instruction with slower sequences...
//
#ifdef __aarch64__
#define int8_bin_op(x) if (x < num_reg / 2) ({ \
    uint8x16_t m = vreinterpretq_u8_u16(vdupq_n_u16(_w[x])); \
    asm ("cmtst %[a].16b, %[a].16b, %[b].16b\n" : [a] "+w"(m), [b] "+w"(mask) ::); \
    tmp##x = vpadalq_u8(tmp##x, m & in0); \
});
#else
#define int8_bin_op(x) if (x < num_reg / 2) ({ \
    uint8x16_t m = vreinterpretq_u8_u16(vdupq_n_u16(_w[x])); \
    m = vtstq_u8(m, mask); \
    tmp##x = vpadalq_u8(tmp##x, m & in0); \
});
#endif

#define readtmp(x) if (_x == x) r = tmp##x;
#define deftmp(x) uint16x8_t tmp##x = {};

#define int8_bin_op2(x) if (x < num_reg) ({ \
    uint16x8_t t = ({ int _x = (x/2); uint16x8_t r; for_each_reg2(readtmp); r; }); \
    v##x = vreinterpretq_u8_u32(vaddw_u16(vreinterpretq_u32_u8(v##x), \
           (x & 1) ? vget_high_u16(t) : vget_low_u16(t))); \
});

#define int8_bin_kernel(in, weight, depth) ({ \
    uint16_t *_i = (void*) (in); \
    uint16_t *_w = (void*) (weight); \
    uint8x16_t mask = {128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1}; \
    uint _d = (depth); \
    int z, i; \
    for (z = 0; z < _d / 2; ) { \
        for_each_reg(deftmp); \
        for (i = 0; i < 128 && z < _d / 2; i++, z++) { \
            uint8x16_t in0 = vreinterpretq_u8_u16(vdupq_n_u16(_i[z])); \
            for_each_reg(int8_bin_op); \
            _w += num_reg / 2; \
        } \
        for_each_reg(int8_bin_op2); \
    } \
})

#ifndef __aarch64__
#define vpaddq_u8(a, b) vcombine_u8( \
    vpadd_u8(vget_low_u8(a), vget_high_u8(a)), \
    vpadd_u8(vget_low_u8(b), vget_high_u8(b)));
#endif

__attribute__ ((always_inline))
static void binarize(uint32_t *output, uint8x16_t *buf_u8, uint size)
{
    uint8x16_t mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
    uint k;

    _assert(size % 2 == 0);

    for (k = 0; k < size; k++)
        buf_u8[k] &= mask;

    for (k = 0; k < size / 2; k++)
        buf_u8[k] = vpaddq_u8(buf_u8[k*2], buf_u8[k*2+1]);

    for (k = 0; k < (size + 3) / 4; k++)
        buf_u8[k] = vpaddq_u8(buf_u8[k*2], buf_u8[k*2+1]);

    for (k = 0; k < (size + 7) / 8; k++)
        buf_u8[k] = vpaddq_u8(buf_u8[k*2], buf_u8[k*2+1]);

    for (k = 0; k < size / 2; k++)
        output[k] = vreinterpretq_u32_u8(buf_u8[k / 4])[k % 4];
}

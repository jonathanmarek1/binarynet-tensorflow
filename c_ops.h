#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*
implementation of the layers in C
definitely needs some cleaning up
*/

typedef unsigned uint;
#define sizearray(x) (sizeof(x)/sizeof(*x))

typedef struct {
    uint dim, shape[4];
    void *data;
    void *storage[2];
    uint type;
} tensor;

//static assert after optimizations
#define _str(x) #x
#define str(x) _str(x)
#define _assert(x) if (!(x)) \
    __asm__("assert fail (expression not constant or false), line=" str(__LINE__) " file="__FILE__);

#define ptr(t, type, x, y...) ({ \
    type _type; \
    __auto_type _t = t; \
    typeof(x) _x[] = {x, y}; \
    _assert(sizearray(_x) == _t.dim); \
    int offset = 0; \
    for (int i = 0; i < sizearray(_x); i++) \
        offset = offset * _t.shape[i] + _x[i]; \
    (typeof(_type)*) (_t.data + _Generic(_type, \
        bool: offset / 8, \
        uint8_t: offset, \
        uint16_t: offset * 2, \
        float : offset * 4)); \
})

#define output(t, x, y...) ({ \
    __auto_type _t = t; \
    typeof(x) _x[] = {x, y}; \
    tensor out; \
    out.dim = sizearray(_x); \
    for (int i = 0; i < sizearray(_x); i++) \
        out.shape[i] = _x[i]; \
    out.storage[0] = _t.storage[0]; \
    out.storage[1] = _t.storage[1]; \
    out.data = _t.data == _t.storage[0] ? _t.storage[1] : _t.storage[0]; \
    out.type = _t.type; \
    out; \
})
enum {
    FLOAT,
    INT8,
    INT16,
    BINARY,
};

enum {
    ACTIVE_NONE,
    ACTIVE_BIN,
    ACTIVE_RELU,
};
#include <arm_neon.h>

#include "c_ops_neon.h"
//#include "cortexa53.h"

__attribute__ ((always_inline))
static void binarize_float(uint32_t *output, float32x4_t *buf, uint size)
{
    uint32x4_t *buf_u32 = (void*) buf;
    uint8x16_t *buf_u8 = (void*) buf;
    uint k;

    _assert(size % 8 == 0);

    //note: for some reason clang stores min as a list of 24 pointers on the stack instead of a single pointer

    //printf("%f %f %f %f\n", buf[0][0], min[0][0], beta[0][0], sum);
    //printf("%f\n", buf[0][0] + min[0][0] * sum);

    for (k = 0; k < size; k++)
        buf_u32[k] = vcltq_f32(buf[k], vdupq_n_f32(0.0f));
        //buf_u32[k] = vcltq_f32(buf[k] + min[k] * vdupq_n_f32(sum), beta[k]);

    for (k = 0; k < size / 4; k++) {
        buf_u8[k] = vcombine_u8(
        vmovn_u16(vcombine_u16(vmovn_u32(buf_u32[k*4+0]), vmovn_u32(buf_u32[k*4+1]))),
        vmovn_u16(vcombine_u16(vmovn_u32(buf_u32[k*4+2]), vmovn_u32(buf_u32[k*4+3]))));
    }

    binarize(output, buf_u8, size / 4);
}

__attribute__ ((always_inline))
static void binarize_u16(uint8x16_t *output, uint16x8_t *buf, uint16x8_t *beta, uint size)
{
    uint8x16_t buf_u8[size / 2];
    uint k;

    _assert(size % 4 == 0);

    for (k = 0; k < size; k++)
        buf[k] = vcltq_u16(beta[k], buf[k]);

    for (k = 0; k < size / 2; k++)
        buf_u8[k] = vcombine_u8(vmovn_u16(buf[k*2]), vmovn_u16(buf[k*2+1]));

    binarize((void*) output, buf_u8, size / 2);
}

//#include <android/log.h>
//#define printf(args...) __android_log_print(ANDROID_LOG_ERROR, "test_app", args)

__attribute__ ((always_inline))
static tensor conv2d(tensor input, tensor filter, tensor out_b, uint sx, uint sy, uint px, uint py, uint activation, float *qp, int *sync)
{
    _assert(input.dim == 3 && filter.dim == 4);

    bool accum16 = (input.type == BINARY && filter.type == BINARY);
#define f(x) register uint8x16_t v##x;
    for_each_reg(f)
#undef f
#if 0 //def __aarch64__
#define v_ptr (uint8x16_t[]) {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23}
    int split_size = accum16 ? 128 : 96;
    int split_num = (filter.shape[3] - 1) / split_size + 1;
    int num_reg_base = accum16 ? 16 : 24;
#else
    int split_size = accum16 ? 64 : 32;
    int split_num = (filter.shape[3] - 1) / split_size + 1;
    int num_reg_base = accum16 ? 8 : 8;
#define v_ptr (uint8x16_t[]) {v0, v1, v2, v3, v4, v5, v6, v7}
#endif
    int num_reg;
    int num_reg_last = (filter.shape[3] / (accum16 ? 8 : 4) - 1) % num_reg_base + 1;

    _assert(filter.shape[3] % (accum16 ? 8 : 4) == 0);

    int i, j, k, x, y, z, u, v, out_w, out_h, xl, yl;
    int32_t sum;

    out_w = (input.shape[0] + px * 2 - filter.shape[0]) / sx + 1;
    out_h = (input.shape[1] + py * 2 - filter.shape[1]) / sy + 1;

    _assert((input.shape[0] + px * 2 - filter.shape[0]) % sx == 0);
    _assert((input.shape[1] + py * 2 - filter.shape[1]) % sy == 0);

    tensor output = output(input, out_w, out_h, filter.shape[3]);

    int kk;
    do {
        kk = __sync_fetch_and_add(sync, 1);
        if (kk >= out_w * out_h * split_num)
            break;

        k = kk % split_num; kk /= split_num;
        j = kk % out_h; kk /= out_h;
        i = kk;

        xl = i * sx - px;
        yl = j * sy - py;
        num_reg = (k + 1 == split_num) ? num_reg_last : num_reg_base;

#define f(x) if (x < num_reg) v##x = (uint8x16_t) {};
        for_each_reg(f)
#undef f

        sum = 0;

        for (u = 0; u < filter.shape[0]; u++) for (v = 0; v < filter.shape[1]; v++) {
            x = xl + u;
            y = yl + v;
            if (x < 0 || x >= input.shape[0] || y < 0 || y >= input.shape[1]) {
               if (input.type == BINARY && filter.type == BINARY) {
#define f(x) if (x < num_reg) v##x = \
    vreinterpretq_u8_u16(vreinterpretq_u16_u8(v##x) + vdupq_n_u16(filter.shape[2] / 2));
                for_each_reg(f)
#undef f
                }
                continue;
            }

            if (input.type == FLOAT && filter.type == FLOAT) {
                float_kernel(ptr(input, float, x, y, 0),
                    //ptr(filter, float, u, v, 0, 0) + k * split_size,
                    ptr(filter, float, u, v, 0, 0) + k * filter.shape[2] * split_size,
                             filter.shape[2]);
            } else if (input.type == FLOAT && filter.type == BINARY) {
                float_bin_kernel(ptr(input, float, x, y, 0),
                ptr(filter, bool, u, v, 0, 0) + k * split_size * filter.shape[2] / 8,
                             filter.shape[2]);
            } else if (input.type == INT8 && filter.type == BINARY) {
                _assert(num_reg % 2 == 0);

                for (z = 0; z < filter.shape[2]; z++)
                    sum += *ptr(input, uint8_t, x, y, z);
                int8_bin_kernel(ptr(input, uint8_t, x, y, 0),
                ptr(filter, bool, u, v, 0, 0) + k * split_size * filter.shape[2] / 8,
                             filter.shape[2]);
            } else if (input.type == INT8 && filter.type == INT8) {
                //sum += 128 * filter.shape[2];
                for (z = 0; z < filter.shape[2]; z++)
                    sum += (int8_t) *ptr(input, uint8_t, x, y, z);

                int8_kernel(ptr(input, uint8_t, x, y, 0),
                ptr(filter, uint8_t, u, v, 0, 0) + k * split_size * filter.shape[2],
                             filter.shape[2]);

            } else if (input.type == BINARY && filter.type == BINARY) {
                bin_kernel(ptr(input, bool, x, y, 0),
                ptr(filter, bool, u, v, 0, 0) + k * split_size * filter.shape[2] / 8,
                             filter.shape[2]);
            } else {
                _assert(0);
            }
        }

        void *out;
        void *b, *m, *c, *d;

        if (input.type == FLOAT || input.type == INT8) { //float accum
            out = ptr(output, float, i, j, k * split_size);
            m = ptr(out_b, float, 0, k * split_size);
            b = ptr(out_b, float, 1, k * split_size);
            c = ptr(out_b, float, 2, k * split_size);
            d = ptr(out_b, float, 3, k * split_size);

            float sum2;
            if (filter.type == INT8 && input.type == INT8)
                sum2 = (float) sum + (filter.shape[0]*filter.shape[1]*filter.shape[2]) * qp[1];

#define f(x) if (x < num_reg) ({ \
    float32x4_t tmp; \
    tmp = vreinterpretq_f32_u8(v##x); \
    if (input.type == INT8) { \
        int32x4_t i = vreinterpretq_s32_u8(v##x); \
        if (filter.type == BINARY) \
            i = -i * vdupq_n_s32(2) + vdupq_n_s32(sum); \
        tmp = vcvtq_f32_s32(i); \
    } \
    if (filter.type == FLOAT) \
        tmp = tmp + ((float32x4_t*)m)[x]; \
    else if (filter.type == INT8 && input.type == INT8) \
        tmp = vmulq_n_f32(tmp + ((float32x4_t*)c)[x] * vdupq_n_f32(sum2) + vdupq_n_f32(qp[1]) * ((float32x4_t*)d)[x], qp[0]) * ((float32x4_t*)b)[x] + ((float32x4_t*)m)[x]; \
    else \
        tmp = tmp * vmulq_n_f32(((float32x4_t*)m)[x], qp[0]) + ((float32x4_t*)b)[x]; \
    if (activation == ACTIVE_RELU) \
        tmp = vmaxq_f32(tmp, vdupq_n_f32(0.0f)); \
    if (activation == ACTIVE_BIN) \
        v##x = tmp; \
    else \
        ((float32x4_t*)out)[x] = tmp; \
});
        for_each_reg(f)
#undef f
            if (activation == ACTIVE_BIN)
                binarize_float((void*) ptr(output, bool, i, j, k * split_size), (void*) v_ptr, num_reg);
        } else if (input.type == BINARY) {
            _assert(filter.type == BINARY);;

            if (activation == ACTIVE_BIN) {
                out = ptr(output, bool, i, j, k * split_size);
                b = ptr(out_b, uint16_t, 0, k * split_size);
                binarize_u16(out, (void*) v_ptr, b, num_reg);
            } else {
                out = ptr(output, float, i, j, k * split_size);
                m = ptr(out_b, float, 0, k * split_size);
                b = ptr(out_b, float, 1, k * split_size);
#if 1
                int z;
                for (z = 0; z < num_reg * 8; z++) {
                    ((float*) out)[z] = (float) (int) (filter.shape[0]*filter.shape[1]*filter.shape[2] - 2 * vreinterpretq_u16_u8(v_ptr[z / 8])[z % 8])
                        * ((float*) m)[z] + ((float*) b)[z];

                    if (activation == ACTIVE_RELU)
                        ((float*) out)[z] = __builtin_fmaxf(((float*) out)[z], 0.0f);
                }

#else
#define f(x) if (x < num_reg) ({ \
    float32x4_t tmp0, tmp1, in_size; \
    in_size = vdupq_n_f32(filter.shape[0]*filter.shape[1]*filter.shape[2]); \
    tmp0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v##x))); \
    tmp1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v##x))); \
    tmp0 = in_size - tmp0 * vdupq_n_f32(2.0f); \
    tmp1 = in_size - tmp1 * vdupq_n_f32(2.0f); \
    tmp0 = tmp0 * ((float32x4_t*)m)[x*2+0] + ((float32x4_t*)b)[x*2+0]; \
    tmp1 = tmp1 * ((float32x4_t*)m)[x*2+1] + ((float32x4_t*)b)[x*2+1]; \
    ((float32x4_t*)out)[x*2+0] = tmp0; \
    ((float32x4_t*)out)[x*2+1] = tmp1; \
});
        for_each_reg(f)
#undef f
#endif
            }
        } else {
            _assert(0);
        }
    } while (1);

    if (activation == ACTIVE_BIN) {
        output.type = BINARY;
    } else {
        output.type = FLOAT;
         //_assert(input.type == FLOAT);
    }

    return output;
}

static __attribute__ ((always_inline))
tensor maxpool(tensor input, uint w, uint h, uint sx, uint sy, void *xor, int *sync)
{
    int out_w, out_h;
    int i, j, k, x, y;

    uint8x16_t *in[w * h], *out;
    uint8x16_t *xorp = xor;

    out_w = (input.shape[0] - w) / sx + 1;
    out_h = (input.shape[1] - h) / sy + 1;

    tensor output = output(input, out_w, out_h, input.shape[2]);

    int kk;
    do {
        kk = __sync_fetch_and_add(sync, 1);
        if (kk >= out_w * out_h)
            break;

        j = kk % out_h; kk /= out_h;
        i = kk;

        if (input.type == BINARY) {

        for (x = 0; x < w; x++)
            for (y = 0; y < h; y++)
                in[x * h + y] = (void*) ptr(input, bool, i * sx + x, j * sy + y, 0);

        out = (void*) ptr(output, bool, i, j, 0);
        if (input.shape[2] % 128 == 0) {
        for (k = 0; k < input.shape[2] / 128; k++) {
            out[k] = in[0][k];
            for (x = 1; x < w * h; x++)
                out[k] &= in[x][k];

            if (xorp)
                out[k] ^= xorp[k];
        }
        } else if (input.shape[2] == 96) {
            uint32_t *ptr = (uint32_t*) in[0];
            uint32x4_t tmp;// = { ptr[0], ptr[1], ptr[2], 0};

            tmp = (uint32x4_t) {ptr[0], ptr[1], ptr[2]};
            for (x = 1; x < w * h; x++) {
                ptr = (uint32_t*) in[x];
                tmp &= (uint32x4_t) {ptr[0], ptr[1], ptr[2]};
            }
            if (xorp)
                tmp ^= vreinterpretq_u32_u8(*xorp);

            ptr = (uint32_t*) out;
            ptr[0] = tmp[0];
            ptr[1] = tmp[1];
            ptr[2] = tmp[2];
        } else {
            _assert(0);
        }
        } else {
        _assert(input.type == FLOAT);
        float32x4_t *in[w * h], *out;
        for (x = 0; x < w; x++)
            for (y = 0; y < h; y++)
                in[x * h + y] = (void*) ptr(input, float, i * sx + x, j * sy + y, 0);

        _assert(input.shape[2] % 4 == 0);
        out = (void*) ptr(output, float, i, j, 0);
        for (k = 0; k < input.shape[2] / 4; k++) {
            out[k] = in[0][k];
            for (x = 1; x < w * h; x++)
                out[k] = vmaxq_f32(out[k], in[x][k]);
        }
        }
    } while (1);

    return output;
}

static __attribute__ ((always_inline))
tensor xnornet_fix(tensor input, uint8_t *xor)
{
    int i, j, k;
    _assert(input.type == BINARY);
    _assert(input.shape[2] % 8 == 0);

    tensor output = output(input, input.shape[0], input.shape[1], input.shape[2]);

    for (i = 0; i < input.shape[0]; i++) for (j = 0; j < input.shape[1]; j++) {
        for (k = 0; k < input.shape[2]; k += 8)
            *(uint8_t*)ptr(output, bool, i, j, k) =
                *(uint8_t*)ptr(input, bool, i, j, k) ^ xor[k / 8];
    }
    return output;
}

static __attribute__ ((always_inline))
tensor quantize(tensor input, float *qparam, bool uint8, bool need_min)
{
    float min = 0.0f, max = 0.0f, m;
    float *in;
    int i;
    uint8_t *out, tmp;

    tensor output = output(input, input.shape[0], input.shape[1], input.shape[2]);
    output.type = INT8;

    in = input.data;
    out = output.data;

    for (i = 0; i < input.shape[0] * input.shape[1] * input.shape[2]; i++) {
        max = __builtin_fmaxf(max, in[i]);
        if (need_min)
            min = __builtin_fminf(min, in[i]);
    }

    qparam[0] = (max - min) / 256.0f;
    m = 256.0f / (max - min);
    qparam[1] = min * m + (uint8 ? 0.0f : 128.0f);

    for (i = 0; i < input.shape[0] * input.shape[1] * input.shape[2]; i++) {
        tmp = __builtin_fminf(__builtin_fmaxf((in[i] - (need_min ? min : 0.0f)) * m, 0.0f), 255.0f);
        out[i] = tmp - (uint8 ? 0 : 128);
    }

    /*_assert(input.dim == 3);
    _assert(input.type == FLOAT);
    _assert(input.shape[0] * input.shape[1] * input.shape[2] % 16 == 0);

    tensor output = output(input, input.shape[0], input.shape[1], input.shape[2]);
    output.type = INT8;

    int i;
    float32x4_t *in;//, *out;
    uint8x16_t *out;
    float32x4_t max[4] = {0}, min[4] = {0}, m, b;
    uint32x4_t u32[4];
    uint16x8_t u16[2];
    uint8x16_t u8;
    float x, _min;

    //todo: max/min initialation when need_min is true

    in = input.data;

    for (i = 0; i < input.shape[0] * input.shape[1] * input.shape[2] / 16; i++) {
        max[0] = vmaxq_f32(max[0], in[0]);
        max[1] = vmaxq_f32(max[1], in[1]);
        max[2] = vmaxq_f32(max[2], in[2]);
        max[3] = vmaxq_f32(max[3], in[3]);

        if (need_min) {
            min[0] = vminq_f32(min[0], in[0]);
            min[1] = vminq_f32(min[1], in[1]);
            min[2] = vminq_f32(min[2], in[2]);
            min[3] = vminq_f32(min[3], in[3]);
        }

        in += 4;
    }

    max[0] = vmaxq_f32(max[0], max[1]);
    max[1] = vmaxq_f32(max[2], max[3]);
    max[0] = vmaxq_f32(max[0], max[1]);
    x = fmaxf(fmaxf(max[0][0], max[0][1]), fmaxf(max[0][2], max[0][3]));
    qparam[0] = x / 256.0f;
    m = vdupq_n_f32(256.0f / x);

    if (need_min) {
        min[0] = vminq_f32(min[0], min[1]);
        min[1] = vminq_f32(min[2], min[3]);
        min[0] = vminq_f32(min[0], min[1]);
        x = fminf(fminf(min[0][0], min[0][1]), fminf(min[0][2], min[0][3]));
        qparam[1] = x;
        b = vdupq_n_f32(x);
    } else {
        qparam[1] = 0.0f;
    }

    in = input.data;
    out = output.data;

    for (i = 0; i < input.shape[0] * input.shape[1] * input.shape[2] / 16; i++) {
        if (!need_min) {
            u32[0] = vcvtq_u32_f32(in[0] * m);
            u32[1] = vcvtq_u32_f32(in[1] * m);
            u32[2] = vcvtq_u32_f32(in[2] * m);
            u32[3] = vcvtq_u32_f32(in[3] * m);
        } else {
            u32[0] = vcvtq_u32_f32((in[0] - b) * m);
            u32[1] = vcvtq_u32_f32((in[1] - b) * m);
            u32[2] = vcvtq_u32_f32((in[2] - b) * m);
            u32[3] = vcvtq_u32_f32((in[3] - b) * m);
        }

        u16[0] = vcombine_u16(vqmovn_u32(u32[0]), vqmovn_u32(u32[1]));
        u16[1] = vcombine_u16(vqmovn_u32(u32[2]), vqmovn_u32(u32[3]));

        u8 = vcombine_u8(vqmovn_u16(u16[0]), vqmovn_u16(u16[1]));
        if (!uint8)
            u8 -= vdupq_n_u8(128);
        *out++ = u8;

        in += 4;
    } */

    return output;
}

#ifdef PRINT_TIME
#include <stdio.h>
#include <time.h>
#define TIME() ({ t1 = get_time(); printf("%f\n", (double) (t1 - t0) / 1000000.0); t0 = t1; })
static uint64_t get_time(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
#else
#define TIME()
#endif

#define w_float_bin_float(x, y, z) struct { float w[(x)*(y)]; float b[y*z]; }
#define w_bin(x, y) struct { uint8_t w[(x)*(y)/8]; uint16_t b[y]; }
#define w_bin_float_bin(x, y) struct { uint8_t w[(x)*(y)/8]; float m[y]; float b[y]; }

#define w_float(x, y) struct { float w[(x)*(y)]; float b[y]; }

#define w_bin_float(x, y) struct { uint8_t w[(x)*(y)/8]; float b[y*2]; }
#define w_int8(x, y) struct { int8_t w[(x)*(y)]; float b[y*4]; }

#include <pthread.h>
#ifndef NUM_THREAD
#define NUM_THREAD 1
#endif
struct thread_arg {
    void *in, *weights, *tmp;
    int *sync;
    uint id;
    int *wait_cnt;
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    float quant_param[2];
};

__attribute__ ((always_inline))
static void wait(struct thread_arg *arg, int i)
{
    pthread_mutex_lock(arg->mutex);
    *arg->wait_cnt += 1;
    if (*arg->wait_cnt < i * NUM_THREAD) {
        pthread_cond_wait(arg->cond, arg->mutex);
    } else {
        *arg->sync = 0;
        pthread_cond_broadcast(arg->cond);
    }
    pthread_mutex_unlock(arg->mutex);
}

static void* worker(void *_arg);

__attribute__ ((always_inline))
void* FUNCTION_NAME(void *in, void *weights, void *tmp) {
    pthread_t thread[NUM_THREAD];
    void *ret;
    int sync = 0, i, wait_cnt = 0;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

    //_assert(((unsigned long) in & 15) == 0);
    //_assert(((unsigned long) tmp & 15) == 0);
    //_assert(((unsigned long) weights & 15) == 0);

    struct thread_arg arg[NUM_THREAD];
    for (i = 0; i < NUM_THREAD; i++) {
        arg[i] = (struct thread_arg) {in, weights, tmp, &sync, i, &wait_cnt, &mutex, &cond};
        pthread_create(&thread[i], 0, worker, &arg[i]);
    }

    for (i = 0; i < NUM_THREAD; i++)
         pthread_join(thread[i], &ret);
    return ret;
}

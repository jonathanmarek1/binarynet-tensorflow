#include "util.h"
#include "xnornet.h"
#include "xnornet_bwn.h"
#include <assert.h>
#include <stdio.h>
#include <sched.h>

/*
clang-5.0 -target aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include -Wno-builtin-requires-header test_xnornet.c util.c xnornet.c xnornet_bwn.c -lm -I. -flto -Ofast -pthread -mcpu=cortex-a53
*/

static uint8_t tmp[xnornet_tmp_size] __attribute__((aligned(16)));
static uint8_t tmp2[xnornet_bwn_tmp_size] __attribute__((aligned(16)));

#include <time.h>
static uint64_t get_time(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
#define TIME(x...) ({ uint64_t t0 = get_time(); (x); uint64_t t1 = get_time(); \
    printf("time: %f\n", (float) (t1 - t0) / 1000000.0f); })

int main(void)
{
    int err;
    string weights, weights2, image;
    float xf[227*227*3], *y;
    int top[5];

    err = sched_setscheduler(getpid(), SCHED_FIFO, &(struct sched_param) {.sched_priority = 1});
    if (err)
        printf("failed to set priority\n");

    err = file_mmap(&weights, "xnornet_weights");
    assert(!err);

    err = file_mmap(&weights2, "xnornet_bwn_weights");
    assert(!err);

    err = file_mmap(&image, "image");
    assert(!err);

    assert(weights.size == xnornet_size);
    assert(weights2.size == xnornet_bwn_size);

    {
        float m[] = {0.01735949, 0.01772787, 0.01774145};
        float b[] = {-2.13645733, -2.04468092, -1.81410977};
        float *ptr = image.ptr;
        for (int i = 0; i < 227*227*3; i++)
            xf[i] = ptr[i] * m[i % 3] + b[i % 3];
    }
    TIME(y = xnornet(xf, weights.ptr, tmp));
    softmax(y, 1000);
    top5(top, y, 1000);

    printf("XNORNET:\n%u:%f\n%u:%f\n%u:%f\n%u:%f\n%u:%f\n",
        top[0], y[top[0]],
        top[1], y[top[1]],
        top[2], y[top[2]],
        top[3], y[top[3]],
        top[4], y[top[4]]);

    TIME(y = xnornet_bwn(xf, weights2.ptr, tmp2));
    softmax(y, 1000);
    top5(top, y, 1000);

    printf("BWN:\n%u:%f\n%u:%f\n%u:%f\n%u:%f\n%u:%f\n",
        top[0], y[top[0]],
        top[1], y[top[1]],
        top[2], y[top[2]],
        top[3], y[top[3]],
        top[4], y[top[4]]);

    return 0;
}

/*
clang-5.0 -target aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include -Wno-builtin-requires-header -mcpu=cortex-a53 benchmark.c benchmark/*.c -I. -flto -Ofast -pthread -DPRINT_TIME
*/
#include <stdint.h>
#include <stdio.h>
#include "benchmark/benchmark_float.h"
#include "benchmark/benchmark_int8.h"
#include "benchmark/benchmark_float_bin.h"
#include "benchmark/benchmark_int8_bin.h"
#include "benchmark/benchmark_bin.h"

static uint8_t data[benchmark_float_size] __attribute__((aligned(16)));
static uint8_t x[1024*1024] __attribute__((aligned(16)));
static uint8_t tmp[1024*1024] __attribute__((aligned(16)));

#include <sched.h>

int main(void)
{
    int r = sched_setscheduler(getpid(), SCHED_FIFO, &(struct sched_param) {.sched_priority = 1});
    printf("sched_setscheduler %i\n", r);

    printf("float:\n");
    benchmark_float(x, data, tmp);
    printf("int8:\n");
    benchmark_int8(x, data, tmp);
    printf("float_bin:\n");
    benchmark_float_bin(x, data, tmp);
    printf("int8_bin:\n");
    benchmark_int8_bin(x, data, tmp);
    printf("bin:\n");
    benchmark_bin(x, data, tmp);
    return 0;
}

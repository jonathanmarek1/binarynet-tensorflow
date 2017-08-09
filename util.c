#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "util.h"

int file_mmap(string *res, char *path)
{
    int fd;
    struct stat stat;
    void *map;

    fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;

    map = fstat(fd, &stat) ?
        MAP_FAILED :
        mmap(0, stat.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED)
        return -1;

    *res = (string) {map, stat.st_size};
    return 0;
}

void top5(int *top, float *y, int num)
{
    int i, j;
    for (i = 0; i < num; i++) {
        for (j = 0; j < 5 && j < i && y[top[j]] >= y[i]; j++);

        if (j == 0) {
            top[4] = top[3];
            top[3] = top[2];
            top[2] = top[1];
            top[1] = top[0];
            top[0] = i;
        }

        if (j == 1) {
            top[4] = top[3];
            top[3] = top[2];
            top[2] = top[1];
            top[1] = i;
        }

        if (j == 2) {
            top[4] = top[3];
            top[3] = top[2];
            top[2] = i;
        }

        if (j == 3) {
            top[4] = top[3];
            top[3] = i;
        }

        if (j == 4) {
            top[4] = i;
        }
    }
}

#include <math.h>

void softmax(float *out, int num)
{
    uint i, j;
    double d, q;

    for (i = 0, d = 0.0, q = -INFINITY; i < num; i++) {
        d += exp((double) out[i] / 1.0);
        if (q < out[i]) {
            q = out[i];
            j = i;
        }
    }

    for (i = 0; i < num; i++)
        out[i] = (exp((double) out[i] / 1.0) / d);
}

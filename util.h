#include <stdlib.h>

typedef struct {
    void *ptr;
    size_t size;
} string;

int file_mmap(string *res, char *path);

void top5(int *top, float *y, int num);
void softmax(float *out, int num);

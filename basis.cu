#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int num)
{
    int i = threadIdx.x;
    if (i < num)
    {
        c[i] = b[i] + a[i];
    }
}

int main(int argc, char const *argv[])
{
    // init data
    const int num = 10;
    int a[num], b[num], c[num];
    int *a_gpu, *b_gpu, *c_gpu;
    for (auto i = 0; i < num; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }
    cudaMalloc((void **)&a_gpu, sizeof(a));
    cudaMalloc((void **)&b_gpu, sizeof(b));
    cudaMalloc((void **)&c_gpu, sizeof(c));

    // copy data
    cudaMemcpy(a_gpu, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(b), cudaMemcpyHostToDevice);

    // do
    add<<<1, num>>>(a_gpu, b_gpu, c_gpu, num);

    cudaMemcpy(c, c_gpu, sizeof(c), cudaMemcpyDeviceToHost);

    // viz
    for (size_t i = 0; i < num; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}

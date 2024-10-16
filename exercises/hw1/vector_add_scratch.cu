#include <stdio.h>

__global__ void vector_add(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        c[idx] = a[idx] + b[idx];
}

#define N 2

int main() {
    int size = sizeof(double) * N;

    double* a = (double*)malloc(size);
    double* b = (double*)malloc(size);
    double* c = (double*)malloc(size);

    a[0] = 0.840188;
    a[1] = 1.34;
    b[0] = 0.394383;
    b[1] = 0.43;

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    vector_add<<<1,N>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i) {
        printf("A[%d] = %f\n", i, a[i]);
        printf("B[%d] = %f\n", i, b[i]);
        printf("C[%d] = %f\n", i, c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
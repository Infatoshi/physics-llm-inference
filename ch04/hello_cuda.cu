// Chapter 04: Hello CUDA - Vector Addition
// This is the simplest CUDA kernel: adding two vectors element-wise.

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

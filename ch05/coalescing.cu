// Chapter 05: Memory Coalescing
// Demonstrates the performance difference between coalesced and strided access

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void coalesced_read(float* data, float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = data[tid] * 2.0f;
    }
}

__global__ void strided_read(float* data, float* out, int n, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * stride;
    if (idx < n) {
        out[tid] = data[idx] * 2.0f;
    }
}

__global__ void coalesced_write(float* data, float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = data[tid] * 2.0f;
    }
}

__global__ void strided_write(float* data, float* out, int n, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * stride;
    if (idx < n && tid < n / stride) {
        out[idx] = data[tid] * 2.0f;
    }
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);
    int stride = 32;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; i++) {
        coalesced_read<<<blocks, threads>>>(d_in, d_out, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        coalesced_read<<<blocks, threads>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalesced_time;
    cudaEventElapsedTime(&coalesced_time, start, stop);

    int strided_blocks = (n / stride + threads - 1) / threads;
    for (int i = 0; i < 10; i++) {
        strided_read<<<strided_blocks, threads>>>(d_in, d_out, n, stride);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        strided_read<<<strided_blocks, threads>>>(d_in, d_out, n, stride);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float strided_time;
    cudaEventElapsedTime(&strided_time, start, stop);

    float coalesced_bw = (2.0f * bytes * 100) / (coalesced_time * 1e-3) / 1e9;
    float strided_bw = (2.0f * bytes / stride * 100) / (strided_time * 1e-3) / 1e9;

    printf("Memory Coalescing Benchmark\n");
    printf("===========================\n");
    printf("Data size: %d MB\n", (int)(bytes / 1024 / 1024));
    printf("\n");
    printf("Coalesced: %.2f ms, %.1f GB/s\n", coalesced_time, coalesced_bw);
    printf("Strided:   %.2f ms, %.1f GB/s\n", strided_time, strided_bw);
    printf("Slowdown:  %.1fx\n", strided_time / coalesced_time);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

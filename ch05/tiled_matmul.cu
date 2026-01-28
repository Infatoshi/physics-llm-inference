// Chapter 05: Tiled Matrix Multiplication
// Demonstrates shared memory tiling for matrix multiplication

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void naive_matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiled_matmul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void benchmark(const char* name,
               void (*kernel)(float*, float*, float*, int, int, int),
               float* A, float* B, float* C, int M, int N, int K,
               dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 100;

    float flops = 2.0f * M * N * K;
    float tflops = flops / (time_ms * 1e-3) / 1e12;

    printf("%s: %.3f ms, %.2f TFLOPS\n", name, time_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix Multiplication Benchmark\n");
    printf("================================\n");
    printf("Size: %dx%d @ %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("\n");

    benchmark("Naive", naive_matmul, d_A, d_B, d_C, M, N, K, grid, block);
    benchmark("Tiled", tiled_matmul, d_A, d_B, d_C, M, N, K, grid, block);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    return 0;
}

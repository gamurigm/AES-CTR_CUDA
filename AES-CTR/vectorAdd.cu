#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1 << 20;  // 1 millón de elementos
    size_t size = n * sizeof(float);

    // Reservar memoria en el host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Inicializar los vectores en el host
    for (int i = 0; i < n; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Reservar memoria en el device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copiar datos del host al device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configurar los bloques y la cuadrícula
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Lanzar el kernel en la GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copiar el resultado de vuelta al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verificar el resultado
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

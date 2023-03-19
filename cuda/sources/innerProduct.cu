#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 10e7 // Size of vectors
#define ceil(x,y) (((x)+(y)-1)/(y))                                             
// Kernel function to compute the inner product of two vectors on GPU
__global__ void innerProductKernel(const float* A, const float* B, float* partialSums) {
    __shared__ float temp[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    temp[threadIdx.x] = A[idx] * B[idx];
    __syncthreads();

    // Reduce within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
    }
    
    // Store partial sum for block
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = temp[0];
    }
}

float innerProductCUDA(const float* A, const float* B) {
    float* d_A;
    float* d_B;
    float* d_partialSums;

    float *partialSums = new float [N/256+1];

    // float partialSums[N / 256 + 1] = {0};
    float h_C = 0;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_partialSums, ceil(N, 256) * sizeof(float));

    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(ceil(N, dimBlock.x), 1, 1);

    auto start_cuda = std::chrono::high_resolution_clock::now();



    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    innerProductKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_partialSums);

    cudaMemcpy(partialSums, d_partialSums, ceil(N, 256) * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ceil(N, dimBlock.x); i++) {
        h_C += partialSums[i];
    }
    auto stop_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda - start_cuda);
    std::cout << "CUDA_REAL: " << duration_cuda.count() <<" microseconds"<<std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partialSums);

    return h_C;
}





float cublas_dot_product(float* x, float* y, int Narr)
{
    // Initialize cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on the device for the vectors
    float* d_x, * d_y;
    cudaMalloc(&d_x, Narr * sizeof(float));
    cudaMalloc(&d_y, Narr * sizeof(float));


    auto start_cublas = std::chrono::high_resolution_clock::now();
    

    // Copy the vectors from host to device
    cudaMemcpy(d_x, x, Narr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, Narr * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the inner product of the vectors using cublas
    float result;
    cublasSdot(handle, Narr, d_x, 1, d_y, 1, &result);
    auto stop_cublas = std::chrono::high_resolution_clock::now();
    auto duration_cublas = std::chrono::duration_cast<std::chrono::microseconds>(stop_cublas - start_cublas);
    std::cout << "CUBLAS: " <<duration_cublas.count()<<" microseconds\n";
    // Free memory on the device
    cudaFree(d_x);
    cudaFree(d_y);

    // Destroy cublas handle
    cublasDestroy(handle);

    return result;
}


// Function to compute the inner product of two vectors using the CPU
float innerProductCPU(const float* A, const float* B) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        result += A[i] * B[i];
    }
    return result;
}

int main() {
    std::vector<float> A(N);
    std::vector<float> B(N);

    // Initialize vectors with random values
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // Compute inner product using CUDA
    auto start_cuda = std::chrono::high_resolution_clock::now();
    float result_cuda = innerProductCUDA(A.data(), B.data());
    auto stop_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda - start_cuda);

    // Compute inner product using CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float result_cpu = innerProductCPU(A.data(), B.data());
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);

    float result_cublas = cublas_dot_product(A.data(), B.data(),N);

    
    
    // Print execution times
    // std::cout << "CUDA: " << duration_cuda.count() << " microseconds\n";
    
    
    std::cout << "CPU: " << duration_cpu.count() << " microseconds\n";  

    // Check if results are equal
    if (result_cpu == result_cuda && result_cpu == result_cublas) {
        std::cout << "Results are equal\n";
    } else {
        std::cout << "Results are not equal\n "<<result_cublas<<std::endl;
        
    }
    return 0;
}

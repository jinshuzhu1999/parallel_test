#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 10e4 // Size of vectors
#define ceil(x,y) (((x)+(y)-1)/(y)) 
// nvcc -o test_cuda test_cuda.cu -lcudart -lcublas && test_cuda

                                    


int main() {
int maxSharedMemPerBlock;
cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
int nShareMem = maxSharedMemPerBlock / sizeof(double);
std::cout<<nShareMem<<std::endl;
}

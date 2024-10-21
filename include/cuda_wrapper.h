#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <cuda_runtime.h>
#include "mpi.h"

struct cudaStream_x{
    MPI_Request *request;
    int d = 0;
};

// Wrapper for cudaMalloc
cudaError_t cudaMalloc(void **devPtr, size_t size);

// Wrapper for cudaFree
cudaError_t cudaFree(void *devPtr);

//Wrapper for Cuda Stream Creation
cudaError_t cudaStreamCreate(cudaStream_x *stream);

//Wrapper for Cuda Stream Synchronization
cudaError_t cudaStreamSynchronize(cudaStream_x stream);

#endif

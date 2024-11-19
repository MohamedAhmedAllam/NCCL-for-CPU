#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include "mpi.h"
#include <cuda_runtime.h>

struct CUstream_st{
    MPI_Request *request;
    int d = 0;
};

//typedef struct CUstream_st *cudaStream_t; //do I need this ?

// Wrapper for cudaMalloc
//cudaError_t cudaMalloc(void **devPtr, size_t size);

// Wrapper for cudaFree
//cudaError_t cudaFree(void *devPtr);

//Wrapper for Cuda Stream Creation
cudaError_t cudaStreamCreate(cudaStream_t *stream);

//Wrapper for Cuda Stream Synchronization
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

#endif

#include "cuda_wrapper.h"

#include <cstdlib>
#include <cstring>
#include <cstdio> 


// Wrapper for cudaMalloc
cudaError_t cudaMalloc(void **devPtr, size_t size){

    if (devPtr == nullptr){
        return cudaErrorInvalidValue;
    }
    *devPtr = malloc(size);

    if (*devPtr == nullptr){
        return cudaErrorMemoryAllocation;
    }

    return cudaSuccess;
}

// Wrapper for cudaFree
cudaError_t cudaFree(void **devPtr, size_t size){
    if (devPtr == nullptr){
        return cudaErrorInvalidValue;
    }

    free(devPtr);
    return cudaSuccess;

}



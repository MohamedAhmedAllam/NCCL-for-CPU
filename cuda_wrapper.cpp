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
cudaError_t cudaFree(void *devPtr){
    if (devPtr == nullptr){
        return cudaErrorInvalidValue;
    }

    free(devPtr);
    return cudaSuccess;

}

//Wrapper for Cuda Stream Creation
cudaError_t cudaStreamCreate(cudaStream_x *stream){
    if (stream == nullptr){
        return cudaErrorInvalidValue;
    }

    stream->request = new MPI_Request;

    //*stream = nullptr;  //Not sure of this but just in case
    return cudaSuccess;

}

//Wrapper for Cuda Stream Synchronization
cudaError_t cudaStreamSynchronize(cudaStream_x stream){
    //currently no stream to synchronize on
    MPI_Wait(stream.request, NULL );
    return cudaSuccess;
}



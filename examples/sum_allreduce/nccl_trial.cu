// nccl_allgather_example.cpp
#include <stdio.h>
#include "nccl.h"
#include "cuda_wrapper.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

const double epsilon = 1e-6;
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(float *a, float *b, float *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}



int main(int argc, char* argv[])
{

  int n = 100000;

  // Host input vectors
  float *h_a;
  float *h_b;
  //Host output vector
  float *h_c;

  // Device input vectors
  float *d_a;
  float *d_b;
  //Device output vector
  float *d_c;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(float);

  // Allocate memory for each vector on host
  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  int i;
  // Initialize vectors on host
  for( i = 0; i < n; i++ ) {
      h_a[i] = sin(i)*sin(i);
      h_b[i] = cos(i)*cos(i);
  }

  // Copy host vectors to device
  cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n/blockSize);

  // Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  // Copy array back to host
  cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

  // Sum up vector c and print result divided by n, this should equal 1 within error
  double sum = 0;
  for(i=0; i<n; i++)
      sum += h_c[i];
  sum/=(float)n;
  if(abs(sum-1.0)<epsilon)
      printf("PASS\n");
  else
      printf("FAIL\n");


  //END OF VECADD
  ////////////////////////////////////////////////
  
  //int size = 32*1024*1024; // Number of elements to gather
  int size = n;

  int myRank, nRanks, localRank = 0;
  
  // Initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  // Calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }
  
  ncclUniqueId id;
  ncclComm_t comm;
  //float *sendbuff, *recvbuff;
  float *recvbuff;
  cudaStream_t s;

  // Get NCCL unique ID at rank 0 and broadcast it to all others
  //if (myRank == 0) {NCCLCHECK(ncclGetUniqueId(&id));}
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Picking a GPU based on localRank and allocating device buffers
  //CUDACHECK(cudaSetDevice(localRank));  
  
  //CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));
  
  // Initialize send buffer
  /*
  for (int i=0; i<size; i++){
    sendbuff[i] = (float)(myRank); // Assign unique values per rank

  }
  */
  printf("[MPI Rank %d] Initiate ---> Send buffer content: First = %.2f, Last = %.2f \n", myRank, d_c[0], d_c[size-1]);

  
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)d_c, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s)); //SUM
  //NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclMin, comm, s)); //MIN
  //NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclMax, comm, s)); //MAX

  printf("[MPI Rank %d] PreSync: ---> Recv buffer content: First = %.2f, Last = %.2f \n", myRank, recvbuff[0], recvbuff[size-1]);

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));


  printf("[MPI Rank %d] Success: ---> Recv buffer content: First = %.2f, Last = %.2f \n", myRank, recvbuff[0], recvbuff[size-1]);

  // Free device buffers
  //CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // Finalize custom NCCL communicator
  NCCLCHECK(ncclCommDestroy(comm));
  
   // Finalizing MPI
  MPICHECK(MPI_Finalize());
  printf("FINALIZED\n");
  

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}


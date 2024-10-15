// nccl_allgather_example.cpp
#include <stdio.h>
#include "cuda_wrapper.h"
#include "nccl.h"
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

extern "C" {
    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank);
    ncclResult_t ncclCommDestroy(ncclComm_t comm);
    ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                               ncclDataType_t datatype, ncclComm_t comm,
                               cudaStream_t stream);
    int getCommD(ncclComm_t comm);
}

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

int main(int argc, char* argv[])
{
  int size = 32*1024*1024; // Number of elements to gather

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
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  // Get NCCL unique ID at rank 0 and broadcast it to all others
  //if (myRank == 0) {NCCLCHECK(ncclGetUniqueId(&id));}
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Picking a GPU based on localRank and allocating device buffers
  //CUDACHECK(cudaSetDevice(localRank));  
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * nRanks * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));
  
  // Initialize send buffer
  for (int i=0; i<size; i++){
    sendbuff[i] = (float)(myRank); // Assign unique values per rank
  }
  printf("[MPI Rank %d] Initiate: %.2f, %.2f \n", myRank, sendbuff[0], sendbuff[size-1]);

  // Initialize custom NCCL communicator with embedded MPI_Comm
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  // Perform AllGather using the NCCL wrapper (internally uses MPI_Allgather)
  NCCLCHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, comm, s));

  // Completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  // Print received data
  printf("[MPI Rank %d] Success: ", myRank);
  for (int i=0; i<nRanks; i++){
    printf("Received Elem No.%d = %.1f & %.1f || ", i, recvbuff[i*size] , recvbuff[(i+1)*size-1]); // Assign unique values per rank
  }
  printf("\n");

  // Free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // Finalize custom NCCL communicator
  NCCLCHECK(ncclCommDestroy(comm));

   // Finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}

// nccl_allgather_example.cpp (Data Parallelism)
#include <stdio.h>
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>


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

__global__ void forward_pass(float *x, float *w, float *y_pred,int N, int n){

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < N){
    float z = 0.0f;
    for (int i=0; i<n; i++){
      z += x[id*n+i]*w[i];
    }

    y_pred[id] = 1.0f / (1.0f+expf(-z));
  }
  
}

__global__ void backward_pass(float *y_pred, float *y_target, float *dldz, int N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < N){
   float dl_dy = 2.0f * (y_pred[id] - y_target[id]);  //RMSE
   float dy_dz = y_pred[id] * (1.0f - y_pred[id]);

   dldz[id] = dl_dy * dy_dz; // Gradient W.R.T wtx

  }
}

__global__ void calc_grads(float *x, float *dldz, float *dw, int N, int n){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < n){
    dw[id] = 0.0f;
   for(int j=0; j<N; j++){
      dw[id] += dldz[j] * x[j*n+id];
   }
  }
}

int main(int argc, char* argv[])
{

  int N = 1024; //BatchSize
  int n = 1000; // weight layer
  float lr = 0.01f;

  //MPI initialization
  int myRank, nRanks, localRank = 0;
  
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
  cudaStream_t s;

  // Get NCCL unique ID at rank 0 and broadcast it to all others
  //if (myRank == 0) {NCCLCHECK(ncclGetUniqueId(&id));}
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Picking a GPU based on localRank and allocating device buffers
  //CUDACHECK(cudaSetDevice(localRank));  
  
  CUDACHECK(cudaStreamCreate(&s));
  
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  // Allocate Host Memory
  float *h_x = (float*)malloc(N * n * sizeof(float));
  float *h_w = (float*)malloc(n*sizeof(float));
  float *h_y_target = (float*)malloc(N*sizeof(float));

  // Initialize input data and weights
  //TODO: this should be done on a single process and shared with the others
  for (int i=0; i<N*n;i++){
    h_x[i] = ((float)rand() / RAND_MAX-1) *2 - 1;
  }
  for (int i=0; i<n;i++){
    h_w[i] = ((float)rand() / RAND_MAX-1) * 0.01f;
  }
  for (int i=0; i<N;i++){
    h_y_target[i] = ((float)rand() / RAND_MAX-1) > 0.5 ? 1.0f : 0.0f;
  }

  //Allocate device memory 
  float *d_x, *d_w, *d_y_target, *d_w_grad, *d_y_pred, *dldz;
  CUDACHECK(cudaMalloc(&d_x, N*n*sizeof(float))); 
  CUDACHECK(cudaMalloc(&d_w, n*sizeof(float))); 
  CUDACHECK(cudaMalloc(&d_y_target, N*sizeof(float))); 
  CUDACHECK(cudaMalloc(&d_w_grad, n*sizeof(float))); 
  CUDACHECK(cudaMalloc(&d_y_pred, N*sizeof(float)));
  CUDACHECK(cudaMalloc(&dldz, N*sizeof(float))); 
 

  //copy data to device
  CUDACHECK(cudaMemcpy(d_x, h_x, N*n*sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_w, h_w, n*sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_y_target, h_y_target, N*sizeof(float), cudaMemcpyHostToDevice));


  int N_LOOPS = 1;    //No of training loops
  int blockSize = 512;
  int gridSize = (N+blockSize-1) / blockSize;

  for (int tl=0; tl<N_LOOPS; tl++){
    //Zero Gradients
    CUDACHECK(cudaMemset(d_w_grad, 0, n*sizeof(float)));  

    //ForwardPass & Synchronize
    forward_pass<<<gridSize, blockSize>>>(d_x, d_w, d_y_pred, N, n);
    CUDACHECK(cudaGetLastError());

    //BackwardPass & Synchronize
    backward_pass<<<gridSize, blockSize>>>(d_y_pred, d_y_target, dldz, N);
    CUDACHECK(cudaGetLastError());

    //TODO Not optimal to loop over N in each kernel (done to avoid AddAtomic)
    calc_grads<<<(n+blockSize-1) / blockSize, blockSize>>>(d_x, dldz, d_w_grad, N, n);
    CUDACHECK(cudaGetLastError());

    printf("before: %.3f_%.3f\n", d_w_grad[0], d_w_grad[n-1]);
    //Gradient Accumulation using NCCL
    NCCLCHECK(ncclAllReduce((const void*)d_w_grad, (void*)d_w_grad, n, ncclFloat, ncclSum, comm, s)); //SUM
    CUDACHECK(cudaStreamSynchronize(s));

    printf("after: %.3f_%.3f\n", d_w_grad[0], d_w_grad[n-1]);
  }

  // Free device buffers
  //CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(d_x));
  CUDACHECK(cudaFree(d_w));
  CUDACHECK(cudaFree(d_w_grad));
  CUDACHECK(cudaFree(d_y_target));
  CUDACHECK(cudaFree(d_y_pred));
  CUDACHECK(cudaFree(dldz));


  free(h_x);
  free(h_w);
  free(h_y_target);

  // Finalize custom NCCL communicator
  NCCLCHECK(ncclCommDestroy(comm));
  
   // Finalizing MPI
  MPICHECK(MPI_Finalize());
  printf("FINALIZED\n");

  return 0;
}


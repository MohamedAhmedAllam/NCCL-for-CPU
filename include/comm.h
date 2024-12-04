#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "mpi.h"

struct ncclComm {
    MPI_Comm mpiComm;
    int d = 0;
};

struct CUstream_st{
    MPI_Request *request;
    int d = 0;
};


#endif

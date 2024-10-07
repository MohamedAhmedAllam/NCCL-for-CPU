#include "nccl.h"

//#include <cstdlib>
//#include <cstring>
//#include <cstdio>
#include "mpi.h"


#include "nccl.h"
#include "mpi.h"

// COPIED from original nccl to map erros strings
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

// Function to map ncclDataType_t to MPI_Datatype
MPI_Datatype getMpiDataType(ncclDataType_t dtype) {
    switch(dtype) {
        case ncclInt8:
            return MPI_CHAR;
        case ncclUint8:
            return MPI_UNSIGNED_CHAR;
        case ncclInt32:
            return MPI_INT;
        case ncclUint32:
            return MPI_UNSIGNED;
        case ncclInt64:
            return MPI_LONG_LONG;
        case ncclUint64:
            return MPI_UNSIGNED_LONG_LONG;
        case ncclFloat16:
            // MPI does not have a native MPI_FLOAT16 type. Use MPI_HALF_FLOAT if available.
            // MPI_HALF_FLOAT is defined in MPI 3.1 and later. Otherwise, consider using MPI_SHORT.
            #ifdef MPI_HALF_FLOAT
                return MPI_HALF_FLOAT;
            #else
                // Fallback or handle as needed
                return MPI_DATATYPE_NULL;
            #endif
        case ncclFloat32:
            return MPI_FLOAT;
        case ncclFloat64:
            return MPI_DOUBLE;
        default:
            // Handle unknown data types
            // You can choose to return MPI_DATATYPE_NULL or handle the error as needed
            return MPI_DATATYPE_NULL;
    }
}

// Function to map ncclRedOp_t to MPI_Op
MPI_Op getMpiRedOp(ncclRedOp_t op) {
    switch(op) {
        case ncclSum:
            return MPI_SUM;
        case ncclProd:
            return MPI_PROD;
        case ncclMin:
            return MPI_MIN;
        case ncclMax:
            return MPI_MAX;
        // see a solution for ncclAvg Op
        default:
            // Handle unknown reduction operations
            return MPI_OP_NULL;
    }
}


ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, MPI_Comm comm, cudaStream_t stream){
    
    MPI_Datatype datatype_mpi = getMpiDataType(datatype);
    if (datatype_mpi == MPI_DATATYPE_NULL){
        return ncclInvalidArgument;  // Sure of this ? it should return an NCCL return
    }

    MPI_Op op_mpi = getMpiRedOp(op);
    if (op_mpi == MPI_OP_NULL){
        return ncclInvalidArgument;  // Sure of this ? it should return an NCCL return
    }

    int res = MPI_Allreduce(sendbuff, recvbuff, count, datatype_mpi, op_mpi, comm);
    if (res != MPI_SUCCESS){
        return ncclSystemError;
    }

    return ncclSuccess;
}
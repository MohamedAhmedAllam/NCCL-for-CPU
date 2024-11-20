#!/bin/bash

export NCCLCPU_PATH=../..

export CuPBoP_PATH=$NCCLCPU_PATH/CuPBoP
export LD_LIBRARY_PATH=$CuPBoP_PATH/build/runtime:$CuPBoP_PATH/build/runtime/threadPool:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda

CXX="clang++-14" && CXXFLAGS="-O2 -std=c++11"
CUDA_HOME="/usr/local/cuda" 
MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi/include"

INCLUDES="-I$CUDA_HOME/include -I$MPI_HOME -I$NCCLCPU_PATH/include"
LIBS="-L$CUDA_HOME/lib64 -lcudart -lmpi_cxx -lmpi"

INCLUDES2="-I$NCCLCPU_PATH/include -I$MPI_HOME"
LIBS2="-lmpi_cxx -lmpi"


# Compile the CUDA wrapper

echo "Compiling cuda_wrapper.cpp..."
$CXX $CXXFLAGS $INCLUDES -c $NCCLCPU_PATH/src/cuda_wrapper.cpp -o cuda_wrapper.o
if [ $? -ne 0 ]; then
    echo "Failed to compile cuda_wrapper.cpp"
    exit 1
fi


echo "Compiling nccl.cpp..."
$CXX $CXXFLAGS $INCLUDES -c $NCCLCPU_PATH/src/nccl.cpp -o nccl.o
if [ $? -ne 0 ]; then
    echo "Failed to compile  nccl.cpp"
    exit 1
fi


# Compile nccl_trial.cu with clang++
echo "Compiling nccl_trial.cu..."
clang++-14 -std=c++11 -I../.. $INCLUDES2 --cuda-path=$CUDA_HOME --cuda-gpu-arch=sm_50 \
    -L$CUDA_HOME/lib64 -save-temps -c nccl_trial.cu -lcudart_static -ldl -lrt -pthread -v || true

# Translate kernel and host
echo "Translating kernel and host..."
$CuPBoP_PATH/build/compilation/kernelTranslator nccl_trial-cuda-nvptx64-nvidia-cuda-sm_50.bc kernel.bc
if [ $? -ne 0 ]; then
    echo "Kernel translation failed."
    exit 1
fi

$CuPBoP_PATH/build/compilation/hostTranslator nccl_trial-host-x86_64-pc-linux-gnu.bc host.bc
if [ $? -ne 0 ]; then
    echo "Host translation failed."
    exit 1
fi

# Compile kernel and host object files
echo "Compiling kernel and host object files..."
llc-14 --relocation-model=pic --filetype=obj kernel.bc
if [ $? -ne 0 ]; then
    echo "Failed to compile kernel object file."
    exit 1
fi

llc-14 --relocation-model=pic --filetype=obj host.bc
if [ $? -ne 0 ]; then
    echo "Failed to compile host object file."
    exit 1
fi

# Link the executable
echo "Linking the final executable..."
g++ -o nccl_trial -fPIC -no-pie $INCLUDES2 -I$CuPBoP_PATH/runtime/threadPool/include \
    -L$CuPBoP_PATH/build/runtime -L$CuPBoP_PATH/build/runtime/threadPool \
    kernel.o host.o nccl.o cuda_wrapper.o -lCPUruntime -lthreadPool -lpthread $LIBS2
if [ $? -ne 0 ]; then
    echo "Linking failed."
    exit 1
fi

# Run the program using mpirun
echo "Running the program with mpirun..."
mpirun -np 4 ./nccl_trial
if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "Script executed successfully."


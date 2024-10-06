# Makefile for NCCL AllReduce Example with CUDA Wrappers

# Compiler
CXX = nvcc

# Compiler Flags
CXXFLAGS = -O2 -std=c++11

# Include and Library Paths
CUDA_HOME = /usr/local/cuda
NCCL_HOME = /media/mohamed/ext_vol1/Learning/NCCL/code/nccl/build
MPI_HOME = /usr/lib/x86_64-linux-gnu/openmpi/include

INCLUDES = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -I$(MPI_HOME) -I./include
LIBS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lnccl -lcudart -lmpi_cxx -lmpi

# Target executable
TARGET = nccl_allreduce_example

# Source files
SRCS = nccl_allreduce_example.cpp cuda_wrapper.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Compile the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LIBS)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

# Run the program
# Usage: make run NP=4
run: $(TARGET)
	@if [ -z "$(NP)" ]; then \
		echo "Please specify the number of processes. Example:"; \
		echo "make run NP=4"; \
		exit 1; \
	fi
	mpirun -np $(NP) ./$(TARGET)

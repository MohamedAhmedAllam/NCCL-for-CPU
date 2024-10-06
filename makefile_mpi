# Makefile for compiling and running MPI C++ programs

# Compiler
MPICXX = mpic++

# Compiler flags
CXXFLAGS = -O2 -Wall

# Target executable
TARGET = allreduce_example

# Source files
SRCS = allreduce_example.cpp

# Default target
all: $(TARGET)

# Compile the target
$(TARGET): $(SRCS)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean up generated files
clean:
	rm -f $(TARGET)

# Run the program with a specified number of processes
# Usage: make run NP=4
run: $(TARGET)
	@if [ -z "$(NP)" ]; then \
		echo "Please specify the number of processes. Example:"; \
		echo "make run NP=4"; \
		exit 1; \
	fi
	mpirun -np $(NP) ./$(TARGET)

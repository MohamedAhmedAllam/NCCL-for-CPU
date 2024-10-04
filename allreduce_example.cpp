#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Each process sets its send_value to its rank + 1
    int send_value = rank + 1;
    int recv_sum = 0;
    printf("%d\n", send_value);
    
    // Perform the Allreduce operation to sum all send_values
    MPI_Allreduce(&send_value, &recv_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Print the result from each process
    std::cout << "Process " << rank << ": The total sum is " << recv_sum << std::endl;

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}

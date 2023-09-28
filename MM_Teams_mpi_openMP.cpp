// run with 1 process only
// mpicxx -o test <name>.cpp -fopenmp -lOpenCL
// mpirun -np 1 ./test

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <CL/cl.h>
#include <omp.h>

#include <chrono>
using namespace std;
using namespace std::chrono;

// set these evenly: 
unsigned int SZ = 6;        // #1
const unsigned int TS = 6;  // #2
unsigned int **A, **B, **C, **D;

void init(unsigned int **&A, unsigned int rows, unsigned int cols, bool initialise);
void print(unsigned int **A, unsigned int rows, unsigned int cols);
void *add(void *block_id);
void *multiply(unsigned int block_id, unsigned int num_rows_per_process_from_A);
void head(int process_rank, int num_processes);
void node(int process_rank, int num_processes);

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
int err;

cl_mem bufA, bufB, bufC;

size_t local[2] = {TS, TS};
size_t global[2] = {SZ, SZ}; // number of rows and cols or basically the number of threads with indices i and j where i is the row and j is the col of the matrix C

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

void setup_openCL_device_context_queue_kernel();
void setup_kernel_memory(unsigned int num_rows_per_process_from_A);
void copy_kernel_args(unsigned int num_rows_per_process_from_A, int rank);
void free_memory();


void init(unsigned int **&A, unsigned int rows, unsigned int cols, bool initialise)
{
    // Allocate memory for the 2D array A (matrix) with 'rows' rows
    // A is passed by reference, so the changes made here will affect the original pointer in the caller function
    A = (unsigned int **)malloc(sizeof(unsigned int *) * rows); // Allocate memory for an array of row pointers

    // Allocate memory for the data (matrix elements) in a contiguous block
    unsigned int *data = (unsigned int *)malloc(sizeof(unsigned int) * rows * cols);

    // Assign each row pointer in A to point to the corresponding row in 'data'
    for (int i = 0; i < rows; i++)
    {
        A[i] = &data[i * cols];
    }

    // If 'initialise' is true, populate the matrix A with random values
    if (initialise)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Fill each element of A with a random integer between 0 and 3 (inclusive)
                A[i][j] = rand() % 4; // Generates any number less than 4
            }
        }
    }
}


void print(unsigned int **A, unsigned int rows, unsigned int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------\n");
}

// In summary, this main function initializes MPI for parallel processing, 
// measures the execution time of the computation, and controls the execution of different parts of the program 
// based on the rank of the current process.
int main(int argc, char **argv)
{
    // Check if a command-line argument for matrix size (SZ) is provided
    if (argc > 1)
        SZ = atoi(argv[1]); // Convert the argument to an integer and assign it to SZ

    // Initialize the MPI library for parallel processing
    MPI_Init(NULL, NULL);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); // Get the total number of processes
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank); // Get the rank of the current process

    // Record the start time of the computation after generating random numbers
    auto start = high_resolution_clock::now();

    if (process_rank == 0)
    {
        // If the current process has rank 0 (the master process), execute the 'head' function
        head(process_rank, num_processes);
    }
    else
    {
        // If the current process has a rank other than 0, execute the 'node' function
        node(process_rank, num_processes);
    }

    // Record the end time of the computation
    auto end = high_resolution_clock::now();

    // Calculate the duration (execution time) of the computation in microseconds
    auto duration = duration_cast<microseconds>(end - start);

    if (process_rank == 0)
    {
        // If the current process has rank 0, print the total execution time
        cout << "\nTotal execution time: " << duration.count() << " microseconds" << endl;
    }

    // Finalize the MPI library and clean up resources
    MPI_Finalize();
}

// This function primarily manages the matrix multiplication and data distribution among MPI processes. 
// It initializes matrices, scatters matrix A to nodes, broadcasts matrix B to all nodes, performs matrix multiplication 
// on assigned blocks & gathers results
void head(int process_rank, int num_processes)
{
    // Initialize matrices A, B, C, and D with random values if this is the head process (rank 0)
    init(A, SZ, SZ, true);
    init(B, SZ, SZ, true);
    init(C, SZ, SZ, false);
    init(D, SZ, SZ, false);

    // Print matrices A and B (optional, for debugging)
    print(A, SZ, SZ);
    print(B, SZ, SZ);

    // Calculate the number of rows each process will receive from matrix A
    unsigned int num_rows_per_process_from_A = SZ / num_processes;

    // Calculate the total number of elements to broadcast from matrix B
    unsigned int num_elements_to_bcast = SZ * SZ;

    // Create a separate send buffer for scattering matrix A to nodes
    unsigned int **send_buffer = (unsigned int **)malloc(num_processes * sizeof(unsigned int *));
    for (int i = 0; i < num_processes; i++)
    {
        send_buffer[i] = (unsigned int *)malloc(num_rows_per_process_from_A * SZ * sizeof(unsigned int));
    }

    if (process_rank == 0)
    {
        // Fill the send buffer with data from matrix A
        for (int i = 0; i < num_processes; i++)
        {
            for (int j = 0; j < num_rows_per_process_from_A * SZ; j++)
            {
                send_buffer[i][j] = A[i * num_rows_per_process_from_A + j / SZ][j % SZ];
            }
        }
    }

    // Scatter matrix A to nodes using the send buffer
    MPI_Scatter(send_buffer[0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, &A[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all nodes
    MPI_Bcast(&B[0][0], num_elements_to_bcast, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication (OpenMP)
    // Calculate the block ID for each process
    unsigned int block_id = process_rank;

    // Perform matrix multiplication on each block
    multiply(block_id, num_rows_per_process_from_A);

    // Gather results from nodes using a temporary buffer (only done by the head process)
    unsigned int **gather_buffer = nullptr;
    if (process_rank == 0)
    {
        gather_buffer = (unsigned int **)malloc(num_processes * sizeof(unsigned int *));
        for (int i = 0; i < num_processes; i++)
        {
            gather_buffer[i] = (unsigned int *)malloc(num_rows_per_process_from_A * SZ * sizeof(unsigned int));
        }
    }

    MPI_Gather(&C[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, gather_buffer[0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // On rank 0, copy the gathered data into matrix C and free the temporary gather buffer
    if (process_rank == 0)
    {
        for (int i = 0; i < num_processes; i++)
        {
            for (int j = 0; j < num_rows_per_process_from_A * SZ; j++)
            {
                C[i * num_rows_per_process_from_A + j / SZ][j % SZ] = gather_buffer[i][j];
            }
        }

        for (int i = 0; i < num_processes; i++)
        {
            free(gather_buffer[i]);
        }
        free(gather_buffer);
    }

    // Free the send buffer
    for (int i = 0; i < num_processes; i++)
    {
        free(send_buffer[i]);
    }
    free(send_buffer);

    // Print the result matrix C (optional, for debugging)
    print(C, SZ, SZ);
}



void node(int process_rank, int num_processes)
{
    // Calculate the number of rows each process will receive from matrix A
    unsigned int num_rows_per_process_from_A = SZ / num_processes;

    // Initialize local matrices A, B, and C with random values
    init(A, num_rows_per_process_from_A, SZ, true);
    init(B, SZ, SZ, true);
    init(C, num_rows_per_process_from_A, SZ, false);

    // Scatter matrix A from the root process to nodes
    MPI_Scatter(NULL, num_rows_per_process_from_A * SZ, MPI_UNSIGNED, &A[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all nodes (same broadcast as the root process)
    MPI_Bcast(&B[0][0], SZ * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication on the assigned block
    multiply(process_rank, num_rows_per_process_from_A);

    // Gather results from nodes (result gathering is done by the root process)
    MPI_Gather(&C[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, NULL, num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}


void *multiply(unsigned int block_id, unsigned int num_rows_per_process_from_A)
{
    // Calculate the starting and ending row indices for the assigned block
    unsigned int start_row = block_id * num_rows_per_process_from_A;
    unsigned int end_row = (block_id + 1) * num_rows_per_process_from_A;

    // Use OpenMP to parallelize the matrix multiplication loop
    #pragma omp parallel for
    for (unsigned int i = start_row; i < end_row; ++i)
    {
        for (unsigned int j = 0; j < SZ; ++j)
        {
            // Initialize the result matrix element to 0
            C[i][j] = 0; 

            // Perform the actual matrix multiplication for the element (i, j)
            for (unsigned int k = 0; k < SZ; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// This function is responsible for setting up the OpenCL environment for matrix multiplication
void setup_openCL_device_context_queue_kernel()
{
    // Obtain the OpenCL device
    device_id = create_device();
    
    cl_int err;
    
    // Create an OpenCL context associated with the device
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    // Build the OpenCL program using the specified kernel file
    program = build_program(context, device_id, "./matrix_mul.cl");

    // Create an OpenCL command queue for the context and device
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    };

    // Create an OpenCL kernel object from the program for matrix multiplication
    kernel = clCreateKernel(program, "multiply_matrices", &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file 

    Creates a program from the source code in the add_numbers.cl file. 
    Specifically, the code reads the file's content into a char array 
    called program_buffer, and then calls clCreateProgramWithSource.
    */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program 

    The fourth parameter accepts options that configure the compilation. 
    These are similar to the flags used by gcc. For example, you can 
    define a macro with the option -DMACRO=VALUE and turn off optimization 
    with -cl-opt-disable.
    */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

// This function is responsible for identifying and accessing an appropriate OpenCL device for computation;
// helps ensure that the program selects a suitable OpenCL device for computation
cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify an OpenCL platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't identify an OpenCL platform");
        exit(1);
    }

    // Access an OpenCL device
    // First, try to get a GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);

    // If no GPU device is found, try to get a CPU device
    if (err == CL_DEVICE_NOT_FOUND)
    {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }

    // Check if a valid OpenCL device was obtained
    if (err < 0)
    {
        perror("Couldn't access any OpenCL devices");
        exit(1);
    }

    return dev;
}

void free_memory()
{
    // Release OpenCL resources to free memory
    clReleaseKernel(kernel);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

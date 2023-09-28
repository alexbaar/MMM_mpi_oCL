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

// set those evenly:
unsigned int SZ = 16;        // 1
const unsigned int TS = 16;  // 2
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
    // Allocate memory for an array of row pointers
    A = (unsigned int **)malloc(sizeof(unsigned int *) * rows); // number of rows

    // Allocate memory for a contiguous block to store the matrix data
    unsigned int *data = (unsigned int *)malloc(sizeof(unsigned int) * rows * cols);

    // Link the row pointers to the corresponding rows in the data block
    for (int i = 0; i < rows; i++)
    {
        A[i] = &data[i * cols];
    }

    // Initialize the matrix with random values if 'initialise' is true
    if (initialise)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Fill each element of the matrix with a random number between 0 and 3
                A[i][j] = rand() % 4; // any number less than 4
            }
        }
    }
}

// print an array
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

void setup_kernel_memory(unsigned int num_rows_per_process_from_A)
{
// Create OpenCL buffers for matrices A, B, and C

    // Create a read-only buffer for matrices A,B, C 
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, num_rows_per_process_from_A * SZ * sizeof(unsigned int), NULL, NULL);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * SZ * sizeof(unsigned int), NULL, NULL);
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, num_rows_per_process_from_A * SZ * sizeof(unsigned int), NULL, NULL);

// Copy data from the host (CPU) to the OpenCL buffers on the device (GPU)

    // Copy the data from the host matrix A to the bufA buffer on the device
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, num_rows_per_process_from_A * SZ * sizeof(unsigned int), &A[0][0], 0, NULL, NULL);

    // Copy the data from the host matrix B to the bufB buffer on the device
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, SZ * SZ * sizeof(unsigned int), &B[0][0], 0, NULL, NULL);

    // Copy the data from the host matrix C to the bufC buffer on the device
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, num_rows_per_process_from_A * SZ * sizeof(unsigned int), &C[0][0], 0, NULL, NULL);
}


void copy_kernel_args(unsigned int num_rows_per_process_from_A, int rank)
{
    // Set kernel arguments

    // Argument 0: Pass the buffer for matrix A to the OpenCL kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufA);

    // Argument 1: Pass the buffer for matrix B to the OpenCL kernel
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufB);

    // Argument 2: Pass the buffer for matrix C to the OpenCL kernel
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufC);

    // Argument 3: Pass the number of rows per process from matrix A to the OpenCL kernel
    clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&num_rows_per_process_from_A);

    // Argument 4: Pass the size of the matrices (SZ) to the OpenCL kernel
    clSetKernelArg(kernel, 4, sizeof(unsigned int), (void *)&SZ);

    // Check for errors when setting kernel arguments
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}




int main(int argc, char **argv)
{
    // Check if a command-line argument is provided to set the matrix size (SZ)
    if (argc > 1)
        SZ = atoi(argv[1]);

    // Initialize MPI
    MPI_Init(NULL, NULL);

    int num_processes;
    // Get the total number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int process_rank;
    // Get the rank of the current MPI process
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Start measuring time here, after generating random numbers
    auto start = high_resolution_clock::now();

    // Determine the role of the current process (head or node)
    if (process_rank == 0)
    {
        // Execute the head function for the root process
        head(process_rank, num_processes);
    }
    else
    {
        // Execute the node function for non-root processes
        node(process_rank, num_processes);
    }

    // Stop measuring time
    auto end = high_resolution_clock::now();
    // Calculate the duration of the execution
    auto duration = duration_cast<microseconds>(end - start);

    // Display the total execution time for the root process
    if (process_rank == 0)
    {
        cout << "\nTotal execution time: " << duration.count() << " microseconds" << endl;
    }

    // Finalize MPI
    MPI_Finalize();
}




void head(int process_rank, int num_processes)
{
    // Initialize matrices A, B, C, and D with random or predefined values
    init(A, SZ, SZ, true);
    init(B, SZ, SZ, true);
    init(C, SZ, SZ, false);
    init(D, SZ, SZ, false);

    // Print matrices A and B (optional, for debugging)
    print(A, SZ, SZ);
    print(B, SZ, SZ);

    // Calculate the number of rows each process will handle from matrix A
    unsigned int num_rows_per_process_from_A = SZ / num_processes;
    // Calculate the total number of elements in matrix B to be broadcasted
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
                // Calculate the correct indices for scattering
                send_buffer[i][j] = A[i * num_rows_per_process_from_A + j / SZ][j % SZ];
            }
        }
    }

    // Scatter matrix A to nodes using the send buffer
    MPI_Scatter(send_buffer[0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, &A[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all nodes
    MPI_Bcast(&B[0][0], num_elements_to_bcast, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication (OpenMP)
    // Calculate the block ID for each process (used for parallelization)
    unsigned int block_id = process_rank;

    // Perform matrix multiplication on each block
    multiply(block_id, num_rows_per_process_from_A);

    // Gather results from nodes using a temporary buffer
    unsigned int **gather_buffer = nullptr;
    if (process_rank == 0)
    {
        gather_buffer = (unsigned int **)malloc(num_processes * sizeof(unsigned int *));
        for (int i = 0; i < num_processes; i++)
        {
            gather_buffer[i] = (unsigned int *)malloc(num_rows_per_process_from_A * SZ * sizeof(unsigned int));
        }
    }

    // Gather the results of matrix multiplication
    MPI_Gather(&C[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, gather_buffer[0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // On rank 0, copy the gathered data back into matrix C
    if (process_rank == 0)
    {
        for (int i = 0; i < num_processes; i++)
        {
            for (int j = 0; j < num_rows_per_process_from_A * SZ; j++)
            {
                // Calculate the correct indices for gathering
                C[i * num_rows_per_process_from_A + j / SZ][j % SZ] = gather_buffer[i][j];
            }
        }

        // Free the temporary gather buffer
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

    // Matrix Initialization
    unsigned int num_rows_per_process_from_A = SZ / num_processes;

    init(A, num_rows_per_process_from_A, SZ, true);
    init(B, SZ, SZ, true);
    init(C, num_rows_per_process_from_A, SZ, false);

    // Scatter A from root to nodes
    MPI_Scatter(NULL, num_rows_per_process_from_A * SZ, MPI_UNSIGNED, &A[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Broadcast B to all nodes
    MPI_Bcast(&B[0][0], SZ * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication using OpenCL
    setup_openCL_device_context_queue_kernel();
    setup_kernel_memory(num_rows_per_process_from_A);
    copy_kernel_args(num_rows_per_process_from_A, process_rank);

    // Execute the OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // Copy the result back from the OpenCL buffer
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, num_rows_per_process_from_A * SZ * sizeof(unsigned int), &C[0][0], 0, NULL, NULL);

    // Clean up OpenCL resources
    free_memory();

    // Gather results from nodes
    MPI_Gather(&C[0][0], num_rows_per_process_from_A * SZ, MPI_UNSIGNED, NULL, num_rows_per_process_from_A * SZ, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // MPI Finalization
    MPI_Finalize();
}


void *multiply(unsigned int block_id, unsigned int num_rows_per_process_from_A)
{
    // Calculate the starting and ending row indices for this block
    unsigned int start_row = block_id * num_rows_per_process_from_A;
    unsigned int end_row = (block_id + 1) * num_rows_per_process_from_A;

    // Use OpenMP to parallelize matrix multiplication across rows
    // #pragma omp parallel for
    for (unsigned int i = start_row; i < end_row; ++i)
    {
        for (unsigned int j = 0; j < SZ; ++j)
        {
            // Initialize the result matrix C element to 0
            C[i][j] = 0;

            // Perform the actual matrix multiplication for the (i, j) element of C
            for (unsigned int k = 0; k < SZ; ++k)
            {
                // Multiply corresponding elements from matrices A and B, then accumulate the result
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}



void setup_openCL_device_context_queue_kernel()
{
    // Obtain the OpenCL device ID
    device_id = create_device();

    cl_int err;

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    // Build the OpenCL program using an external source file (matrix_mul.cl)
    program = build_program(context, device_id, "./matrix_mul.cl");

    // Create a command queue to enqueue OpenCL commands
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    }

    // Create an OpenCL kernel with the name "multiply_matrices"
    kernel = clCreateKernel(program, "multiply_matrices", &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error = %d", err);
        exit(1);
    }
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
    // First, try to obtain a GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        // If no GPU is found, try to obtain a CPU device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    {
        perror("Couldn't access any OpenCL devices");
        exit(1);
    }

    // Return the chosen OpenCL device
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


// credit: https://github.com/andadiana/cannon-algorithm-mpi/blob/master/CannonMatrixMultiplication/CannonMatrixMultiplication.cpp

// CannonMatrixMultiplication.cpp : Defines the entry point for the console application.
//

// Include necessary header files
//#include "stdafx.h"  // (Optional) Include a precompiled header if used
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib>



using namespace std;
using namespace std::chrono;



// Function to allocate memory for a 2D integer matrix
int allocMatrix(int*** mat, int rows, int cols) {
    // Allocate contiguous memory for the matrix elements
    int* p = (int*)malloc(sizeof(int*) * rows * cols);
    if (!p) {
        return -1;
    }
    // Allocate memory for row pointers
    *mat = (int**)malloc(rows * sizeof(int*));
    if (!mat) {
        free(p);
        return -1;
    }

    // Set up the row pointers to access the contiguous memory
    for (int i = 0; i < rows; i++) {
        (*mat)[i] = &(p[i * cols]);
    }
    return 0;
}

// Function to free memory allocated for a 2D integer matrix
int freeMatrix(int*** mat) {
    free(&((*mat)[0][0]));  // Free the contiguous memory
    free(*mat);             // Free the array of row pointers
    return 0;
}

// Function to perform matrix multiplication of two 2D matrices 'a' and 'b'
// and store the result in 'c'
void matrixMultiply(int **a, int **b, int rows, int cols, int ***c) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int val = 0;
            for (int k = 0; k < rows; k++) {
                val += a[i][k] * b[k][j];
            }
            (*c)[i][j] = val;
        }
    }
}

// Function to print a 2D integer matrix of size 'size'
void printMatrix(int **mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
}

// Function to print a 2D integer matrix to a file
void printMatrixFile(int **mat, int size, FILE *fp) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(fp, "%d ", mat[i][j]);
        }
        fprintf(fp, "\n");
    }
}

int main(int argc, char* argv[]) {

    MPI_Comm cartComm;       // MPI communicator for the Cartesian grid
    int dim[2], period[2], reorder;  // Arrays to specify grid dimensions, periodicity, and reordering
    int coord[2], id;        // Arrays to store coordinates and process rank within the grid
    FILE *fp;                // File pointer for file I/O
    int **A = NULL, **B = NULL, **C = NULL;  // Pointers for 2D integer matrices A, B, and C
    int **localA = NULL, **localB = NULL, **localC = NULL;  // Pointers for local blocks of matrices
    int **localARec = NULL, **localBRec = NULL;  // Pointers for local blocks of received matrices
    int rows = 0;            // Number of rows in the matrices
    int columns;             // Number of columns in the matrices
    int count = 0;           // Counter for reading matrix elements from a file
    int worldSize;           // Total number of MPI processes
    int procDim;             // Dimension of the Cartesian grid
    int blockDim;            // Size of a block in the Cartesian grid
    int left, right, up, down;  // Process ranks in the Cartesian grid for shifting
    int bCastData[4];        // Data for broadcasting grid and matrix dimensions


    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the total number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get the rank of the current MPI process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check if the current process is rank 0 (the master process)
    if (rank == 0) {
        int n;
        char ch;

        // Determine matrix dimensions by reading matrix A from a file
        fp = fopen("A.txt", "r");
        if (fp == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);  // Abort MPI if the file cannot be opened
        }
        while (fscanf(fp, "%d", &n) != EOF) {
            ch = fgetc(fp);
            if (ch == '\n') {
                rows = rows + 1;
            }
            count++;
        }
        columns = count / rows;

        // Check if the number of MPI processes is a perfect square
        double sqroot = sqrt(worldSize);
        if ((sqroot - floor(sqroot)) != 0) {
            printf("[ERROR] Number of processes must be a perfect square!\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        int intRoot = (int)sqroot;
        if (columns % intRoot != 0 || rows % intRoot != 0) {
            printf("[ERROR] Number of rows/columns not divisible by %d!\n", intRoot);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        procDim = intRoot;
        blockDim = columns / intRoot;

        fseek(fp, 0, SEEK_SET);

        // Allocate memory for matrices A and B
        if (allocMatrix(&A, rows, columns) != 0) {
            printf("[ERROR] Matrix allocation for A failed!\n");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        if (allocMatrix(&B, rows, columns) != 0) {
            printf("[ERROR] Matrix allocation for B failed!\n");
            MPI_Abort(MPI_COMM_WORLD, 5);
        }

        // Read matrix A from the file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                fscanf(fp, "%d", &n);
                A[i][j] = n;
            }
        }
        printf("A matrix:\n");
        printMatrix(A, rows);
        fclose(fp);

        // Read matrix B from the file
        fp = fopen("B.txt", "r");
        if (fp == NULL) {
            return 1;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                fscanf(fp, "%d", &n);
                B[i][j] = n;
            }
        }
        printf("B matrix:\n");
        printMatrix(B, rows);
        fclose(fp);

        // Allocate memory for matrix C (result of multiplication)
        if (allocMatrix(&C, rows, columns) != 0) {
            printf("[ERROR] Matrix allocation for C failed!\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }

        // Prepare broadcast data to be sent to all processes
        bCastData[0] = procDim;
        bCastData[1] = blockDim;
        bCastData[2] = rows;
        bCastData[3] = columns;
    }
    auto start = high_resolution_clock::now();
    // Create a 2D Cartesian grid of processes
    MPI_Bcast(&bCastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
    procDim = bCastData[0];
    blockDim = bCastData[1];
    rows = bCastData[2];
    columns = bCastData[3];

    dim[0] = procDim; dim[1] = procDim;
    period[0] = 1; period[1] = 1;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);

    // Allocate local blocks for matrices A and B
    allocMatrix(&localA, blockDim, blockDim);
    allocMatrix(&localB, blockDim, blockDim);

    // Create datatype to describe the subarrays of the global array
    int globalSize[2] = { rows, columns };
    int localSize[2] = { blockDim, blockDim };
    int starts[2] = { 0,0 };
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, blockDim * sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *globalptrA = NULL;
    int *globalptrB = NULL;
    int *globalptrC = NULL;
    if (rank == 0) {
        globalptrA = &(A[0][0]);
        globalptrB = &(B[0][0]);
        globalptrC = &(C[0][0]);
    }

    // Scatter the arrays to all processors
    int* sendCounts = (int*)malloc(sizeof(int) * worldSize);
    int* displacements = (int*)malloc(sizeof(int) * worldSize);

    if (rank == 0) {
        for (int i = 0; i < worldSize; i++) {
            sendCounts[i] = 1;
        }
        int disp = 0;
        for (int i = 0; i < procDim; i++) {
            for (int j = 0; j < procDim; j++) {
                displacements[i * procDim + j] = disp;
                disp += 1;
            }
            disp += (blockDim - 1) * procDim;
        }
    }

    // Scatter matrix A to all processes
    MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
        rows * columns / (worldSize), MPI_INT,
        0, MPI_COMM_WORLD);

    // Scatter matrix B to all processes
    MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
        rows * columns / (worldSize), MPI_INT,
        0, MPI_COMM_WORLD);

    // Allocate memory for local result matrix localC
    if (allocMatrix(&localC, blockDim, blockDim) != 0) {
        printf("[ERROR] Matrix allocation for localC in rank %d failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    // Initial data skew
    MPI_Cart_coords(cartComm, rank, 2, coord);
    MPI_Cart_shift(cartComm, 1, coord[0], &left, &right);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(cartComm, 0, coord[1], &up, &down);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);

    // Initialize localC to zero
    for (int i = 0; i < blockDim; i++) {
        for (int j = 0; j < blockDim; j++) {
            localC[i][j] = 0;
        }
    }

    int** multiplyRes = NULL;
    if (allocMatrix(&multiplyRes, blockDim, blockDim) != 0) {
        printf("[ERROR] Matrix allocation for multiplyRes in rank %d failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 8);
    }

    // Perform the matrix multiplication and update localC
    for (int k = 0; k < procDim; k++) {
        matrixMultiply(localA, localB, blockDim, blockDim, &multiplyRes);

        for (int i = 0; i < blockDim; i++) {
            for (int j = 0; j < blockDim; j++) {
                localC[i][j] += multiplyRes[i][j];
            }
        }

        // Shift matrices localA and localB (left and up)
        MPI_Cart_shift(cartComm, 1, 1, &left, &right);
        MPI_Cart_shift(cartComm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);
    }

    // Gather results from all processes to the master process
    MPI_Gatherv(&(localC[0][0]), rows * columns / worldSize, MPI_INT,
        globalptrC, sendCounts, displacements, subarrtype,
        0, MPI_COMM_WORLD);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Free memory allocated for localC and multiplyRes
    freeMatrix(&localC);
    freeMatrix(&multiplyRes);

    // If this is the master process (rank 0), print the result matrix C
    if (rank == 0) {
        printf("C is:\n");
        printMatrix(C, rows);

        cout << "\nTotal execution time: " << duration.count() << " microseconds" << endl;
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <mpi.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

const int N = 4;

void generateRandomArray(vector<vector<int>>& array, int size)
{
    // Random Number Generation
    // random_device rd;
    // the above implementation of std::random_device did not appear so random. I used the current time as seed (below) instead to give better random values.

    mt19937 e2(time(nullptr));

    // Initialize array with non-zero values; if we skip this, there will be a zero value appearing in the sorted array, which 'steals' the spot of a number
    // from the initial array; we initialize with any number outside of the random number generator, here the range was set between -40000000 and 40000000.
    // so we can use numbers under the lower limit or over the upper limit from line 33
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            array[i][j] = {999}; // Set to a large negative value (guaranteed to be outside the range)
        }
    }

    // Initialize array with random values
    uniform_int_distribution<int> dist(1, 4);

    // Generate random values between -40000000 and 40000000 and assign them to A[i]
    // the number from line 29 will be overwritten
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            array[i][j] = dist(e2);
        }
    }
}

void displayArray(const vector<int>& array, int arraySize)
{
    for (int i = 0; i < arraySize; i++)
    {
        cout << array[i] << "  ";
    }
    cout << endl;
}

void display2DArray(const vector<vector<int>>& array)
{
    for (const vector<int>& row : array)
    {
        for (int value : row)
        {
            cout << value << "  ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {
    // MPI step 1:

    int numtasks, rank, name_len, tag = 1;
    char name[MPI_MAX_PROCESSOR_NAME];

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of tasks/processes
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find the processor name
    MPI_Get_processor_name(name, &name_len);

    int local_size = N / numtasks;
    if (rank == numtasks - 1) {
    local_size = N - (numtasks - 1) * local_size; // Adjust for the last process
}

    // vector of vectors = 2D matrices
    // allocate memory for three 2D vector of vectors (dynamic 2D array)
    vector<vector<int>> A(N, vector<int>(N, 0));
    vector<vector<int>> B(N, vector<int>(N, 2));

    vector<vector<int>> C(N, vector<int>(N, 0));
    // for performing addition locally - no racing condition as we dont try accessing a shared var
    vector<vector<int>> local_A(N, vector<int>(N));
    vector<vector<int>> local_B(N, vector<int>(N));
    vector<vector<int>> local_C(N, vector<int>(N)); 


    // Initialize both A and B consistently across all processes

    int counter = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = counter++;
        }A
    }    

    int counter2 = 3;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = counter2++;
        }
    }

    // MPI step 2: 2D to 1D
    // Scatter & Gather works with a contiguous block of memory, like 1D array. When we have a 2D array (vec of vec) it is not guaranteed.
    // we need to use a 1D array to store and send data in the correct format
    vector<int> flat_A(N * N);
    vector<int> flat_B(N * N);
    vector<int> flat_C(N * N);

    // master
    // put all A, B 2D vector values as one flat array. This way we can use scatter/gather
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                flat_A[i * N + j] = A[i][j];
                flat_B[i * N + j] = B[i][j];
            }
        }
        // check if flattened correctly
        cout << "\nflat A: " << endl;
        displayArray(flat_A, N * N);
        cout << "\nflat B: " << endl;
        displayArray(flat_B, N * N);
    }

    vector<int> local_flat_A(local_size * N);
    vector<int> local_flat_B(local_size * N);
    vector<int> local_flat_C(local_size * N);
    MPI_Barrier(MPI_COMM_WORLD);
    // use scatter to distribute work among all worker processes
    MPI_Scatter(flat_A.data(), local_size * N, MPI_INT, local_flat_A.data(), local_size * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(flat_B.data(), local_size * N, MPI_INT, local_flat_B.data(), local_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    // MPI step 2: convert 1D arrays back to 2D to carry out the multiplication
    for (int i = 0; i < local_size; ++i) {
        for (int j = 0; j < N; ++j) {
            local_A[i][j] = local_flat_A[i * N + j];
            local_B[i][j] = local_flat_B[i * N + j];
        }
    }

    cout << "\nlocal_A:" << endl;
    for (int i = 0; i < local_size; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << local_A[i][j] << "  ";
        }
        cout << endl;
    }

    cout << "\nlocal_B:" << endl;
    for (int i = 0; i < local_size; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << local_B[i][j] << "  ";
        }
        cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Perform local matrix multiplication on each process
    auto start = high_resolution_clock::now();
#pragma omp parallel for collapse(2) 
    for (int i = 0; i < local_size; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i][j] += local_A[i][k] * local_B[k][j];
            }
        }
    }

    cout << "\n\n\nlocal C after multipl:" << endl;
    display2DArray(local_C);

    MPI_Barrier(MPI_COMM_WORLD);
    // MPI step 3: the local result array (local_C) is 2D; Scatter/ Gather only works with 1-dimensional arrays, so
    //             we convert 2D back to 1D
    for (int i = 0; i < local_size; ++i) {
        for (int j = 0; j < N; ++j) {
            local_flat_C[i * N + j] = local_C[i][j];
        }
    }
    cout << "\nflat C:" << endl;
    displayArray(local_flat_C, N * N);
    // use gather to send flat_C result back to the root process
    MPI_Gather(local_flat_C.data(), local_size * N, MPI_INT, flat_C.data(), local_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // MPI step 4: the result arrat (flat_C) is 1D, so again we convert that into 2D array, on the root process
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i][j] = flat_C[i * N + j];
            }
        }
    }
    // Output results on root process
    if (rank == 0) {
        cout << "\nMatrix A:" << endl;
        display2DArray(A);

        cout << "\nMatrix B:" << endl;
        display2DArray(B);

        cout << "\nResult (Matrix C):" << endl;
        display2DArray(C);

        cout << "\nTotal execution time: " << duration.count() << " microseconds" << endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>

using namespace std;
using namespace std::chrono;

const int N = 10;

void generateRandomArray(vector<vector<int>>& array, int size)
{
	// Random Number Generation
    // random_device rd;
	// the above implementation of std::random_device did not appear so random. I used the current time as seed (below) instead to give better random values.

    mt19937 e2(time(nullptr));

    // Initialize array with non-zero values; if we skip this, there will be a zero value appearing in the sorted array, which 'steals' the spot of a number 
	// from the initial array; we initialize with any number outside of the random number generator, here the range was set between -40000000 and 40000000.
	// so we can use numbers under the lower linit or over the upper limit from line 33
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
		array[i][j] = {999}; // Set to a large negative value (guaranteed to be outside the range)
	    }
    }

	// Initialize array with random values
    uniform_int_distribution<int> dist(-100, 100);

	// Generate random values between -40000000 and 40000000 and assign them to A[i]
	// the number from line 29 will be overwritten
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            array[i][j] = dist(e2);
        }
    }


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

int main() {

    // vector of vectors = 2D matrices
    // allocate memory for three 2D vector of vectors (dynamic 2D array)
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));


    generateRandomArray(A, N);
    generateRandomArray(B, N);
    
    // Multiply the matrices locally on each process
    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j]; // Accumulate the product
            }
        }
    }


    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // Save  to a file
/*    ofstream outputFile("results.txt");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << " " << C[i][j] << " ";
        }
        outputFile << endl;
    }
    outputFile.close();
*/    

    cout << "Matrix A:" << endl;
    display2DArray(A);

    cout << "\nMatrix B:" << endl;
    display2DArray(B);

    cout << "\nResult (Matrix C):" << endl;
    display2DArray(C);



    return 0;
}

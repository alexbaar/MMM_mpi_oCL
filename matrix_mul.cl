// using unsigned int to make sure that with larger matrices which elements have higher values, the matrix C can accomodate the result 
// without data overflow

__kernel void multiply_matrices(__global unsigned int* A, __global unsigned int* B, __global unsigned int* C, const unsigned int sz) {
    unsigned int global_id_x = get_global_id(0);
    unsigned int global_id_y = get_global_id(1);
    unsigned int sum = 0;

    for (unsigned int k = 0; k < sz; ++k) {
        sum += A[global_id_x * sz + k] * B[k * sz + global_id_y];
    }

    C[global_id_x * sz + global_id_y] = sum;
}


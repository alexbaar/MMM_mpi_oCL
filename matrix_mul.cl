__kernel void multiply_matrices(__global int* A, __global int* B, __global int* C, const int sz) {
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int sum = 0;

    for (int k = 0; k < sz; ++k) {
        sum += A[global_id_x * sz + k] * B[k * sz + global_id_y];
    }

    C[global_id_x * sz + global_id_y] = sum;
}

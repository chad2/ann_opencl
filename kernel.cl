
__kernel void myGEMM1(const int M, const int N, const int K,
                                      const __global float* A,
                                      const __global float* B,
                                      __global float* C) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
    C[globalCol*M + globalRow] = acc;
}


__kernel void myGEMM4(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = (2/1)*get_group_id(0) + row; // Row ID of C (0..M/WIDTH)
    const int globalCol = 2*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[2][2/1];
    __local float Bsub[2][2/1];

    // Initialise the accumulation registers
    float acc = 0.0f;
    const int numTiles = K/2;
    for (int tile=0; tile<numTiles; tile++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (2/1)*tile + row;
        const int tiledCol = 2*tile + col;
        Asub[col][row] = A[tiledCol*(M/1) + globalRow];
        Bsub[col][row] = B[globalCol*(K/1) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        float vecA, vecB;
        float valB;
        for (int k=0; k<2/1; k++) {
            vecB = Bsub[col][k];
            for (int w=0; w<1; w++) {
                vecA = Asub[1*k + w][row];
                valB = vecB;
                acc += vecA * valB;
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[globalCol*(M/1) + globalRow] = acc;
}


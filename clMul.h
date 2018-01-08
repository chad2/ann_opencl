#ifndef UNTITLED3_CLMUL_H
#define UNTITLED3_CLMUL_H
#define TS 1
#include <CL/cl.h>
#include <iostream>

class clMul {
public:
    clMul();
    virtual ~clMul();
    void cl_mul_mat(float* A, float* B, float* C, int K, int M, int N);

private:
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_event event = NULL;
};


#endif //UNTITLED3_CLMUL_H

#ifndef UNTITLED3_CLMUL_H
#define UNTITLED3_CLMUL_H
#define TS 1
#include <CL/cl.h>
#include <iostream>

class clMul {
public:
    clMul(const char *path = "kernel.cl");
    virtual ~clMul();
    void cl_mul_mat(float* A, float* B, float* C, int K, int M, int N);

private:
    static char ** readCode(const char *path, cl_uint& count);
    static void freeCode(char **code, const size_t length);

    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_event event = NULL;
};


#endif //UNTITLED3_CLMUL_H

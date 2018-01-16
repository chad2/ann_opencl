#ifndef CLMUL_H
#define CLMUL_H
#define WIDTH 1
#include <CL/cl.h>
#include <iostream>

class clMul {
public:
    clMul(const char *path = "kernel.cl");
    virtual ~clMul();
    void cl_mul_mat(float* A, float* B, float* C, int K, int M, int N);
    size_t getWorkGroupSize();
    
    static const char *getOpenCLError(cl_int error);
    static void checkError(cl_int error, const size_t &line);

private:
    static char ** readCode(const char *path, cl_uint& count);
    static void freeCode(char **code, const size_t length);

    size_t wotkGroupSize = 1;
    cl_kernel kernel;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_event event = NULL;
};


#endif //CLMUL_H

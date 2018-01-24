#ifndef CLMUL_H
#define CLMUL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>

#define WIDTH   1
#define KERNEL_FUNCTION_NAME    "myGEMM4"

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

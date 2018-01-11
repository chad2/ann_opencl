// adapted from https://github.com/cnugteren/myGEMM

#include "clMul.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory>

clMul::clMul(const char *path) {
    cl_uint count = 0;
    char **kernelstring = clMul::readCode(path, count);
                
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);

    // Compile the kernel
    program = clCreateProgramWithSource(context, 1, (const char **)kernelstring, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    clMul::freeCode(kernelstring, count);
    kernelstring = nullptr;

    kernel = clCreateKernel(program, "myGEMM4", NULL);
}

clMul::~clMul() {
    // Clean-up OpenCL
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
}

void clMul::cl_mul_mat(float* A, float* B, float* C, int K, int M, int N) {
    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, NULL);
    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);
    // Configure the kernel and set its arguments
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

    //const size_t local[2] = { TS, TS };
    //const size_t global[2] = { M, N };

    const size_t local[2] = { TS/WIDTH, TS };
    const size_t global[2] = { (size_t)(M/WIDTH), (size_t)N };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

    // Wait for calculations to be finished
    clWaitForEvents(1, &event);

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);
    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}

void clMul::freeCode(char **code, const size_t length) {
	if(code == NULL || length < 0) {
		fprintf(stderr, "ERROR: %s - argument is NULL", __FUNCTION__);
		return;
	}
	
	for(size_t i=0; i < length; i++) {
		delete[] code[i];
		code[i] = NULL;
	}
	delete[] code;
	code = NULL;	
}

char ** clMul::readCode(const char *path, cl_uint& count) {
	if(path == NULL) {
		fprintf(stderr, "ERROR: %s - argument is NULL\n", __FUNCTION__);
		return NULL;
	}
	
	char **result = NULL;
	std::unique_ptr<FILE, decltype(&fclose)> fp(fopen(path, "r"), &fclose);
	long lSize = 0;
	count = 0;
	
	if(!fp) {
		fprintf(stderr, "ERROR: %s - failed to open file %s\n", __FUNCTION__, path);
		return result;
	}
	fseek(fp.get(), 0L, SEEK_END);
	lSize = ftell(fp.get());
	rewind(fp.get());
	
	result = new char*[1];
	count = 1;
	result[0] = new char[(lSize+1) * sizeof(char)];
	result[0][lSize] = '\0';
	
	if(1 != fread(result[0], lSize, sizeof(char), fp.get())) {
		fprintf(stderr, "ERROR: %s - fread failed\n", __FUNCTION__);
		freeCode(result, count);
		count = 0;
		return NULL;
	}
	
	
	return result;
}

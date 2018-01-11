// adapted from https://github.com/cnugteren/myGEMM

#include "clMul.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory>

clMul::clMul(const char *path) {
    cl_int err = CL_SUCCESS;

    cl_uint count = 0;
    char **kernelstring = clMul::readCode(path, count);
                
    cl_platform_id platform = 0;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, __LINE__);
    cl_device_id device = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(err, __LINE__);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, __LINE__);
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, __LINE__);
    char deviceName[1024];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    checkError(err, __LINE__);

    // Compile the kernel
    program = clCreateProgramWithSource(context, 1, (const char **)kernelstring, NULL, &err);
    checkError(err, __LINE__);

    //Build kernel with defined options
    char compilerOptions[30] = "-DTS=2 -DWIDTH=2";
    snprintf(compilerOptions, 30, "-DTS=%d -DWIDTH=%d", TS, WIDTH);
    compilerOptions[29] = '\0';
    err = clBuildProgram(program, 0, NULL, compilerOptions, NULL, NULL);
    checkError(err, __LINE__);

    // Check for compilation errors
    size_t logSize;
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkError(err, __LINE__);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    checkError(err, __LINE__);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    clMul::freeCode(kernelstring, count);
    kernelstring = nullptr;

    kernel = clCreateKernel(program, "myGEMM4", &err);
    checkError(err, __LINE__);
}

clMul::~clMul() {
    // Clean-up OpenCL
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
}

void clMul::cl_mul_mat(float* A, float* B, float* C, int K, int M, int N) {
    cl_int err = CL_SUCCESS;

    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(float), NULL, &err);
    checkError(err, __LINE__);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(float), NULL, &err);
    checkError(err, __LINE__);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, &err);
    checkError(err, __LINE__);

    // Copy matrices to the GPU
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(float), A, 0, NULL, NULL);
    checkError(err, __LINE__);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(float), B, 0, NULL, NULL);
    checkError(err, __LINE__);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);
    checkError(err, __LINE__);

    // Configure the kernel and set its arguments
    err = clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    checkError(err, __LINE__);
    err = clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    checkError(err, __LINE__);
    err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    checkError(err, __LINE__);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    checkError(err, __LINE__);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    checkError(err, __LINE__);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);
    checkError(err, __LINE__);

    //const size_t local[2] = { TS, TS };
    //const size_t global[2] = { M, N };

    const size_t local[2] = { TS/WIDTH, TS };
    const size_t global[2] = { (size_t)(M/WIDTH), (size_t)N };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    checkError(err, __LINE__);

    // Wait for calculations to be finished
    err = clWaitForEvents(1, &event);
    checkError(err, __LINE__);

    // Copy the output matrix C back to the CPU memory
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);
    checkError(err, __LINE__);

    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}

void clMul::checkError(cl_int error, const size_t &line) {
    if(error != CL_SUCCESS) {
        std::cerr << "-- Error at " << line << ": " << clMul::getOpenCLError(error) << std::endl;
        throw;
    }
}

const char *clMul::getOpenCLError(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
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

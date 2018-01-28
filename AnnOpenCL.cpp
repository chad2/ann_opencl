#include "AnnOpenCL.h"
#include "clMul.h"
#include "Reader.h"
#include <sstream>
#include <iomanip>
#ifdef DEBUG
#include <chrono>
#endif

AnnOpenCL::AnnOpenCL(
		const int train_size,
		const int test_size,
		const int batchsize,
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int epochs,
		const string path_to_kernel,
		const string train_label_file,
		const string train_image_file,
		const string test_label_file,
		const string test_image_file
) : train_size(train_size),
	test_size(test_size),
	batchsize(batchsize),
	classes(classes),
	first_layer_neurons(first_layer_neurons),
	second_layer_neurons(second_layer_neurons),
	epochs(epochs),
	image_size(IMAGE_SIZE),
	path_to_kernel(path_to_kernel),
	train_label_file(train_label_file),
	train_image_file(train_image_file),
	test_label_file(test_label_file),
	test_image_file(test_image_file)
{
	train_data = Reader::load_data(train_size, train_label_file, train_image_file);
	test_data = Reader::load_data(test_size, test_label_file, test_image_file);

	prepareKernel();
	prepareData();
	setKernelArguments();
}

AnnOpenCL::~AnnOpenCL() {
	AnnOpenCL::free_mat(x_cpu);
	x_cpu = nullptr;
	AnnOpenCL::free_mat(probs_cpu);
	probs_cpu = nullptr;

	delete[] test_data;
	test_data = nullptr;
	delete[] train_data;
	train_data = nullptr;

	clReleaseMemObject(x_cl);
	clReleaseMemObject(w1);
	clReleaseMemObject(b1);
	clReleaseMemObject(r_w1);
	clReleaseMemObject(r_b1);
	clReleaseMemObject(r_a1);
	clReleaseMemObject(w2);
	clReleaseMemObject(b2);
	clReleaseMemObject(r_w2);
	clReleaseMemObject(r_b2);
	clReleaseMemObject(er);
	clReleaseMemObject(probs_cl);
	clReleaseMemObject(d_r_b2);
	clReleaseMemObject(d_w2);
	clReleaseMemObject(d_b2);
	clReleaseMemObject(d_r_a1);
	clReleaseMemObject(d_r_b1);
	clReleaseMemObject(d_w1);
	clReleaseMemObject(d_b1);
	clReleaseMemObject(labels);
	clReleaseMemObject(loss_cl);

	clReleaseKernel(kernel_forward);
	clReleaseKernel(kernel_backprop);
	clReleaseKernel(kernel_backprop_1);
	clReleaseKernel(kernel_backprop_2);
	clReleaseKernel(kernel_backprop_3);
	clReleaseKernel(kernel_update_params);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseProgram(program);

#ifdef DEBUG
	size_t fw_time_average = 0;
	size_t bp_time_average = 0;
	size_t up_time_average = 0;

	for(auto const& i: forward_pass_time) {
		fw_time_average += i;
	}
	fw_time_average = fw_time_average / forward_pass_time.size();
	cout << "forward_pass average time (msec): " << fw_time_average << endl;

	for(auto const& i: backprop_time) {
		bp_time_average += i;
	}
	bp_time_average = bp_time_average / backprop_time.size();
	cout << "backprop average time (msec): " << bp_time_average << endl;

	for(auto const& i: update_params_time) {
		up_time_average += i;
	}
	up_time_average = up_time_average / update_params_time.size();
	cout << "update_params average time (msec): " << up_time_average << endl;
#endif
}

string AnnOpenCL::readKernel(const string path) {
	ifstream ifs(path);

	return string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

void AnnOpenCL::prepareKernel() {
	cl_int err = CL_SUCCESS;
	const string kernel_string = AnnOpenCL::readKernel(this->path_to_kernel);

	cl_platform_id platform = 0;
	err = clGetPlatformIDs(1, &platform, NULL);
	clMul::checkError(err, __LINE__);
	cl_device_id device = 0;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	clMul::checkError(err, __LINE__);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	clMul::checkError(err, __LINE__);
	queue = clCreateCommandQueue(context, device, 0, &err);
	clMul::checkError(err, __LINE__);
	char deviceName[1024];
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
	clMul::checkError(err, __LINE__);

	// Compile the kernel
	const char* kernelstr[] = {kernel_string.c_str()};
	program = clCreateProgramWithSource(context, 1, kernelstr, NULL, &err);
	clMul::checkError(err, __LINE__);

	//Build kernel with defined options
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clMul::checkError(err, __LINE__);

	// Check for compilation errors
	size_t logSize;
	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	clMul::checkError(err, __LINE__);
	char messages[logSize + 1];
	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
	clMul::checkError(err, __LINE__);
	messages[logSize] = '\0';
	if (logSize > 10) {
		cout << ">>> Compiler message: " << endl << messages << endl;
	}

	kernel_forward = clCreateKernel(program, KERNEL_FORWARD, &err);
	clMul::checkError(err, __LINE__);

	kernel_backprop = clCreateKernel(program, KERNEL_BACKPROP, &err);
	clMul::checkError(err, __LINE__);

	kernel_backprop_1 = clCreateKernel(program, KERNEL_BACKPROP_1, &err);
	clMul::checkError(err, __LINE__);

	kernel_backprop_2 = clCreateKernel(program, KERNEL_BACKPROP_2, &err);
	clMul::checkError(err, __LINE__);

	kernel_backprop_3 = clCreateKernel(program, KERNEL_BACKPROP_3, &err);
	clMul::checkError(err, __LINE__);

	kernel_update_params = clCreateKernel(program, KERNEL_UPDATE_PARAMS, &err);
	clMul::checkError(err, __LINE__);
}

void AnnOpenCL::prepareData() {
	cl_int err = CL_SUCCESS;
	float **temp = nullptr;
	size_t h=0, w=0;

	h = batchsize;
	w = image_size*image_size;
	x_cpu = create_mat(h, w);
	x_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------

	//set w1
	h = image_size*image_size;
	w = first_layer_neurons;
	temp = create_mat(h, w);
	AnnOpenCL::init_rand(temp, h, w, -0.1, 0.1);
	w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);
	err = clEnqueueWriteBuffer(queue, w1, CL_TRUE, 0, h*w*sizeof(float), *temp, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);
	AnnOpenCL::free_mat(temp);
	temp = nullptr;

	//set d_w1
	h = image_size*image_size;
	w = first_layer_neurons;
	d_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set r_w1
	h = batchsize;
	w = first_layer_neurons;
	r_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set b1
	h = 1;
	w = first_layer_neurons;
	temp = create_mat(h, w);
	AnnOpenCL::init_rand(temp, h, w, -0.1, 0.1);
	b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);
	err = clEnqueueWriteBuffer(queue, b1, CL_TRUE, 0, h*w*sizeof(float), *temp, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);
	AnnOpenCL::free_mat(temp);
	temp = nullptr;

	//set d_b1
	h = batchsize;
	w = first_layer_neurons;
	d_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set r_b1
	h = batchsize;
	w = first_layer_neurons;
	r_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set d_r_b1
	h = batchsize;
	w = first_layer_neurons;
	d_r_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set r_a1
	h = batchsize;
	w = first_layer_neurons;
	r_a1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set d_r_a1
	h = batchsize;
	w = first_layer_neurons;
	d_r_a1 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set w2
	h = first_layer_neurons;
	w = second_layer_neurons;
	temp = create_mat(h, w);
	AnnOpenCL::init_rand(temp, h, w, -0.1, 0.1);
	w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);
	err = clEnqueueWriteBuffer(queue, w2, CL_TRUE, 0, h*w*sizeof(float), *temp, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);
	AnnOpenCL::free_mat(temp);
	temp = nullptr;

	//set d_w2
	h = first_layer_neurons;
	w = second_layer_neurons;
	d_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set r_w2
	h = batchsize;
	w = second_layer_neurons;
	r_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set b2
	h = 1;
	w = second_layer_neurons;
	temp = create_mat(h, w);
	AnnOpenCL::init_rand(temp, h, w, -0.1, 0.1);
	b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);
	err = clEnqueueWriteBuffer(queue, b2, CL_TRUE, 0, h*w*sizeof(float), *temp, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);
	AnnOpenCL::free_mat(temp);
	temp = nullptr;

	//set d_b2
	h = 1;
	w = second_layer_neurons;
	d_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set r_b2
	h = batchsize;
	w = second_layer_neurons;
	r_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//set d_r_b2
	h = batchsize;
	w = second_layer_neurons;
	d_r_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set er
	h = batchsize;
	w = classes;
	er = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------


	//set probs_cpu
	//set probs_cl
	h = batchsize;
	w = classes;
	probs_cpu = create_mat(h, w);
	probs_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, h*w*sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------

	h = batchsize;
	labels = clCreateBuffer(context, CL_MEM_READ_ONLY, h*sizeof(int), NULL, &err);
	clMul::checkError(err, __LINE__);

	loss_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
	clMul::checkError(err, __LINE__);
}

void AnnOpenCL::setKernelArguments() {
	cl_int err = CL_SUCCESS;
	int arg_pos = 0;

	cl_kernel *kernel_arr[] = {
		&kernel_forward,
		&kernel_backprop,
		&kernel_backprop_1,
		&kernel_backprop_2,
		&kernel_backprop_3,
		&kernel_update_params
	};
	for(int i=0; i < 6; i++) {
		arg_pos = 0;

		// Configure the kernel and set its arguments
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&batchsize);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&classes);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&first_layer_neurons);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&second_layer_neurons);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&epochs);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(int), (void*)&image_size);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(float), (void*)&learning_rate);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&loss_cl);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&x_cl);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&w1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_w1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&r_w1);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&b1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_b1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&r_b1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_r_b1);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&r_a1);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_r_a1);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&w2);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_w2);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&r_w2);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&b2);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_b2);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&r_b2);
		clMul::checkError(err, __LINE__);
		arg_pos++;
		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&d_r_b2);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&er);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&probs_cl);
		clMul::checkError(err, __LINE__);
		arg_pos++;

		err = clSetKernelArg(*(kernel_arr[i]), arg_pos, sizeof(cl_mem), (void*)&labels);
		clMul::checkError(err, __LINE__);
		arg_pos++;
	}
}

float AnnOpenCL::forward_pass(const bool training, const int step) {
#ifdef DEBUG
	auto start = chrono::steady_clock::now();
#endif
	cl_int err = CL_SUCCESS;
	const int data_size = training ? train_size : test_size;
	const imageLabel* data = training ? train_data : test_data;
	size_t h = 0, w = 0;
	float loss = 0;

	//write input to GPU buffer
	h = batchsize;
	w = image_size*image_size;
	load_input(x_cpu, data, batchsize, step * batchsize, data_size);
	err = clEnqueueWriteBuffer(queue, x_cl, CL_TRUE, 0, h*w*sizeof(float), *x_cpu, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------
	const size_t local_forward[] = {static_cast<size_t>(batchsize)};
	const size_t global_forward[] = {static_cast<size_t>(batchsize)};
	clEnqueueNDRangeKernel(queue, kernel_forward, 1, 0, global_forward, local_forward, 0, NULL, &event);

	// Wait for calculations to be finished
	err = clWaitForEvents(1, &event);
	clMul::checkError(err, __LINE__);

	//load loss from GPU
	err = clEnqueueReadBuffer(queue, loss_cl, CL_TRUE, 0, sizeof(float), &loss, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);

#ifdef DEBUG
	auto end = chrono::steady_clock::now();
	forward_pass_time.push_back(chrono::duration_cast<chrono::milliseconds>(end-start).count());
#endif

	return loss;
}

void AnnOpenCL::backprop(const bool training, const int step) {
#ifdef DEBUG
	auto start = chrono::steady_clock::now();
#endif
	cl_int err = CL_SUCCESS;
	const int data_size = training ? train_size : test_size;
	const imageLabel* data = training ? train_data : test_data;
	size_t w = 0;

	//write label for training backpropagation to GPU buffer
	w = batchsize;
	int *temp_labels = new int[w];
	load_labels(temp_labels, data, batchsize, step * batchsize, data_size);
	err = clEnqueueWriteBuffer(queue, labels, CL_TRUE, 0, w*sizeof(int), temp_labels, 0, NULL, NULL);
	delete[] temp_labels;
	temp_labels = nullptr;
	clMul::checkError(err, __LINE__);

	//--------------------------------------------------------------------------------
	/*
	const size_t local_backprop[] = {static_cast<size_t>(batchsize)};
	const size_t global_backprop[] = {static_cast<size_t>(batchsize)};
	clEnqueueNDRangeKernel(queue, kernel_backprop, 1, 0, global_backprop, local_backprop, 0, NULL, &event);
	*/

	const size_t local_backprop_1[] = {static_cast<size_t>(batchsize)};
	const size_t global_backprop_1[] = {static_cast<size_t>(batchsize)};
	clEnqueueNDRangeKernel(queue, kernel_backprop_1, 1, 0, global_backprop_1, local_backprop_1, 0, NULL, &event);

	// Wait for calculations to be finished
	err = clWaitForEvents(1, &event);
	clMul::checkError(err, __LINE__);


	const size_t local_backprop_2[] = {static_cast<size_t>(first_layer_neurons)};
	const size_t global_backprop_2[] = {static_cast<size_t>(first_layer_neurons)};
	clEnqueueNDRangeKernel(queue, kernel_backprop_2, 1, 0, global_backprop_2, local_backprop_2, 0, NULL, &event);

	const size_t local_backprop_3[] = {static_cast<size_t>((image_size*image_size))};
	const size_t global_backprop_3[] = {static_cast<size_t>((image_size*image_size))};
	clEnqueueNDRangeKernel(queue, kernel_backprop_3, 1, 0, global_backprop_3, local_backprop_3, 0, NULL, &event);

	// Wait for calculations to be finished
	err = clWaitForEvents(1, &event);
	clMul::checkError(err, __LINE__);

#ifdef DEBUG
	auto end = chrono::steady_clock::now();
	backprop_time.push_back(chrono::duration_cast<chrono::milliseconds>(end-start).count());
#endif
	//170-180 msec
}

void AnnOpenCL::update_params(const float learning_rate) {
#ifdef DEBUG
	auto start = chrono::steady_clock::now();
#endif
	cl_int err = CL_SUCCESS;

	if(this->learning_rate != learning_rate) {
		this->learning_rate = learning_rate;

		const int arg_pos = 6;
		err = clSetKernelArg(kernel_update_params, arg_pos, sizeof(float), (void*)&this->learning_rate);
		clMul::checkError(err, __LINE__);
	}

	//--------------------------------------------------------------------------------
	const size_t local_update[] = {1};
	const size_t global_update[] = {4};
	clEnqueueNDRangeKernel(queue, kernel_update_params, 1, 0, global_update, local_update, 0, NULL, &event);

	// Wait for calculations to be finished
	err = clWaitForEvents(1, &event);
	clMul::checkError(err, __LINE__);

#ifdef DEBUG
	auto end = chrono::steady_clock::now();
	update_params_time.push_back(chrono::duration_cast<chrono::milliseconds>(end-start).count());
#endif
}

float AnnOpenCL::calc_acc(const bool training, const int step, const bool visual) {
	const int data_size = training ? train_size : test_size;
	imageLabel* data = training ? train_data : test_data;
	const int offset = step*batchsize;
	float acc;
	int correct = 0;
	cl_int err = CL_SUCCESS;

	//load probs from GPU
	err = clEnqueueReadBuffer(queue, probs_cl, CL_TRUE, 0, batchsize*classes*sizeof(float), *probs_cpu, 0, NULL, NULL);
	clMul::checkError(err, __LINE__);

	for(int i=0; i<batchsize; i++){
		int max_index = 0;
		for(int j=0; j<classes; j++){
			if(probs_cpu[i][j]>probs_cpu[i][max_index]){ // find class with highest probability
				max_index = j;
			}
		}
		if(data[(i+offset)%data_size].label==max_index){    // compare to correct class
			correct++;
		}
	}
	acc = (float)correct/batchsize;
	if(visual){   // visualize tested examples
		forward_pass(false, 0);  // starting test sample from index 0
		for(int i=0; i<10 && i<batchsize; i++){
			std::cout << "L: " << +data[(i+offset)%data_size].label << ";  ";
			for(int j=0; j<classes; j++){
				std::cout << j << ": " << std::setprecision(3) << std::fixed << probs_cpu[i][j] << "  ";
			}
			imageLabel::print(&data[(i+offset)%data_size]);
			std::cout << std::endl;
		}
	}
	return acc;
}

float **AnnOpenCL::create_mat(const size_t h, const size_t w) {
    float** ptr = new float*[h]; 		// pointer
    float* pool = new float[h*w];		// mempool
    for(size_t i=0;i<h;i++, pool += w){
        ptr[i] = pool;
    }
    return ptr;
}

void AnnOpenCL::free_mat(float **toFree) {
    if(toFree == nullptr) {
        return;
    }
    delete[] toFree[0];
    delete[] toFree;
}

void AnnOpenCL::init_rand(float **mat, const size_t h, const size_t w, const float min, const float max) {
    for(size_t i=0; i<h; i++){
        for(size_t j=0; j<w; j++){
            mat[i][j] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2*max)));
        }
    }
}

void AnnOpenCL::load_input(float **x, const imageLabel *il, const int amount, const int offset, const int data_size) {
    for(int i=0; i<amount; i++){
        for(int j=0; j<(image_size*image_size); j++){
            x[i][j] = il[(i+offset)%data_size].pixel[j/image_size][j%image_size];
            x[i][j] = (float)x[i][j]/255;  // map to 0-1
        }
    }
}

void AnnOpenCL::load_labels(int *label_target, const imageLabel *il, const int amount, const int offset, const int data_size) {
    for(int i=0; i<amount; i++){
    	label_target[i] = (int) il[(i+offset)%data_size].label;
    }
}

#ifndef ANNOPENCL_H_
#define ANNOPENCL_H_

using namespace std;

#include "imageLabel.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <string>

#define KERNEL_FORWARD			"forward_pass"
#define KERNEL_BACKPROP			"backprop"
#define KERNEL_UPDATE_PARAMS	"update_params"

class AnnOpenCL {
public:
	AnnOpenCL(
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
	);
	virtual ~AnnOpenCL();

	float forward_pass(const bool training, const int step);
	void backprop(const bool training, const int step);
	void update_params(const float learning_rate);
	float calc_acc(const bool training, const int step, const bool visual);

protected:
	static string readKernel(const string path);
	void prepareKernel();
	void prepareData();
	void setKernelArguments();
	void load_input(float **x, const imageLabel *il, const int amount, const int offset, const int data_size);
	void load_labels(int *label_target, const imageLabel *il, const int amount, const int offset, const int data_size);
	static float **create_mat(const size_t h, const size_t w);
	static void free_mat(float **toFree);
	static void init_rand(float **mat, const size_t h, const size_t w, const float min, const float max);

	const int train_size;
	const int test_size;
	const int batchsize;
	const int classes;
	const int first_layer_neurons;
	const int second_layer_neurons;
	const int epochs;
	const int image_size;
	const string path_to_kernel;
	const string train_label_file;
	const string train_image_file;
	const string test_label_file;
	const string test_image_file;

	float learning_rate = 0;

	imageLabel* train_data;
	imageLabel* test_data;

	cl_event event;

	cl_kernel kernel_forward;
	cl_kernel kernel_backprop;
	cl_kernel kernel_update_params;
	cl_context context;
	cl_command_queue queue;
	cl_program program;

	float** x_cpu;      // input matrix
	cl_mem x_cl;

	cl_mem w1;     // weights    -  first layer  -
	cl_mem b1;     // bias                       -
	cl_mem r_w1;   // result x*w1                -
	cl_mem r_b1;   // result r_w1+b1             -
	cl_mem r_a1;   // result activation of r_b1  -
	cl_mem d_r_a1; //            -  first layer gradients  -
	cl_mem d_r_b1; //                                      -
	cl_mem d_w1;   //                                      -
	cl_mem d_b1;   //                                      -

	cl_mem w2;     // weights   -  second layer  -
	cl_mem b2;     // bias                       -
	cl_mem r_w2;   // result r_a1*w2             -
	cl_mem r_b2;   // result r_w2+b2             -
	cl_mem d_r_b2; //           -  second layer gradients  -
	cl_mem d_w2;   //                                      -
	cl_mem d_b2;   //                                      -

	cl_mem er;     // exponated output from network for later use

	cl_mem probs_cl;
	float** probs_cpu;  // resulting probabilities

	cl_mem labels;

	cl_mem loss_cl;
};

#endif /* ANNOPENCL_H_ */

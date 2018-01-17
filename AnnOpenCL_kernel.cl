//                   N     
//                o-----o  
//                |     |  
//              K | [B] |  
//                |     |  
//                o-----o  
//        K          N     
//    o-------o   o-----o  
//  M |  [A]  | M | [C] |  
//    |       |   |     |  
//    o-------o   o-----o  
void mul_mat(__global float *A, __global float *B, __global float *C,
		const int M, const int K, const int N)
{
	for(int i=0; i < M; i++) {
		for(int j=0; j < N; j++) {
			float sum = 0;
			for(int k=0; k < K; k++) {
				sum += A[(i*K) + k] * B[(k*N) + j];
			}
			C[(i*M) + j] = sum;
		}
	}
}

void broadcasted_add(__global float *mat, __global float *bias, __global float *res,
		const int w)
{
	for(int j=0; j < w; j++) {
		res[j] = mat[j] + bias[j];
	}
}

void act_mat(__global float *in, __global float *res, const int w)
{
	for(int j=0; j<w; j++){
		res[j] = (in[j] > 0) ? in[j] : 0;
	}
}

#define FLT_MAX 	0x1.fffffep127f
void exp_mat(__global float *in, __global float *res, const int w)
{
	for(int j=0; j<w; j++){
		res[j] = exp(in[j]);
		if(isinf(res[j])){  // exp result to big - clamp to max result
			res[j] = FLT_MAX;
		}
	}
}

void softmax_mat(__global float *in, __global float *res, const int w)
{
	float sum = 0;
	for(int j=0; j < w; j++) {
		sum += in[j];
	}
	for(int j=0; j < w; j++) {
		res[j] = in[j] / sum;
	}
}

void forward_pass(
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int image_size,
		__global float *x,
		__global float *w1,
		__global float *r_w1,
		__global float *b1,
		__global float *r_b1,
		__global float *r_a1,
		__global float *w2,
		__global float *r_w2,
		__global float *b2,
		__global float *r_b2,
		__global float *er,
		__global float *probs_cl
) {
	const int global_id = get_global_id(0);

	mul_mat(
			&x[global_id * image_size * image_size],
			w1,
			&r_w1[global_id * first_layer_neurons],
			1,
			image_size*image_size,
			first_layer_neurons
	);
	broadcasted_add(
			&r_w1[global_id * first_layer_neurons],
			b1,
			&r_b1[global_id * first_layer_neurons],
			first_layer_neurons
	);
	act_mat(
			&r_b1[global_id * first_layer_neurons],
			&r_a1[global_id * first_layer_neurons],
			first_layer_neurons
	);
	mul_mat(
			&r_a1[global_id * first_layer_neurons],
			w2,
			&r_w2[global_id * second_layer_neurons],
			1,
			first_layer_neurons,
			second_layer_neurons
	);
	broadcasted_add(
			&r_w2[global_id * second_layer_neurons],
			b2,
			&r_b2[global_id * second_layer_neurons],
			second_layer_neurons
	);

	exp_mat(
			&r_b2[global_id * second_layer_neurons],
			&er[global_id * classes],
			classes
	);
	softmax_mat(
			&er[global_id * classes],
			&probs_cl[global_id * classes],
			classes
	);
}

__kernel void kernel_main (
		const int batchsize,
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int epochs,
		const int image_size,
		__global float *x,
		__global float *w1,
		__global float *d_w1,
		__global float *r_w1,
		__global float *b1,
		__global float *d_b1,
		__global float *r_b1,
		__global float *d_r_b1,
		__global float *r_a1,
		__global float *d_r_a1,
		__global float *w2,
		__global float *d_w2,
		__global float *r_w2,
		__global float *b2,
		__global float *d_b2,
		__global float *r_b2,
		__global float *d_r_b2,
		__global float *er,
		__global float *probs_cl
) {
	forward_pass(
		classes,
		first_layer_neurons,
		second_layer_neurons,
		image_size,
		x,
		w1,
		r_w1,
		b1,
		r_b1,
		r_a1,
		w2,
		r_w2,
		b2,
		r_b2,
		er,
		probs_cl
	);
}
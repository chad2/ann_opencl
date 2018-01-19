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
void mul_mat(const __global float *A, const __global float *B, __global float *C,
		const int M, const int K, const int N)
{
	for(int i=0; i < M; i++) {
		for(int j=0; j < N; j++) {
			float sum = 0;
			for(int k=0; k < K; k++) {
				sum += A[(i*K) + k] * B[(k*N) + j];
			}
			C[(i*N) + j] = sum;
		}
	}
}
void mul_mat_transpA(const __global float *A, const __global float *B, __global float *C,
		const int M, const int K, const int N)
{	
	for(int i=0; i < M; i++) {
		for(int j=0; j < N; j++) {
			float sum = 0;
			for(int k=0; k < K; k++) {
				sum += A[(k*M) + i] * B[(k*N) + j];
			}
			C[(i*N) + j] = sum;
		}
	}
}
void mul_mat_transpB(const __global float *A, const __global float *B, __global float *C,
		const int M, const int K, const int N)
{	
	for(int i=0; i < M; i++) {
		for(int j=0; j < N; j++) {
			float sum = 0;
			for(int k=0; k < K; k++) {
				sum += A[(i*K) + k] * B[(j*K) + k];
			}
			C[(i*N) + j] = sum;
		}
	}
}

void broadcasted_add(const __global float *mat, const __global float *bias, __global float *res,
		const int w)
{
	for(int j=0; j < w; j++) {
		res[j] = mat[j] + bias[j];
	}
}

void act_mat(const __global float *in, __global float *res, const int w)
{
	for(int j=0; j<w; j++){
		res[j] = (in[j] > 0) ? in[j] : 0;
	}
}

#define FLT_MAX 	0x1.fffffep127f
void exp_mat(const __global float *in, __global float *res, const int w)
{
	for(int j=0; j<w; j++){
		res[j] = exp(in[j]);
		if(isinf(res[j])){  // exp result to big - clamp to max result
			res[j] = FLT_MAX;
		}
	}
}

void softmax_mat(const __global float *in, __global float *res, const int w)
{
	float sum = 0;
	for(int j=0; j < w; j++) {
		sum += in[j];
	}
	for(int j=0; j < w; j++) {
		res[j] = in[j] / sum;
	}
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

#define FLT_MIN 	0x1.0p-126f
float ce_loss(const __global int *labels, const __global float *a, const int h, const int w)
{
	float sum = 0;
	float val = 0;
	for(int i=0; i < h; i++) {
		val = -log(a[(i*w) + labels[i]]);
		if(a[(i*w) + labels[i]] == 0) {
			val = -log(FLT_MIN);
		}
		sum += val;
	}
	
	return sum;
}

__kernel void forward_pass(
		const int batchsize,
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int epochs,
		const int image_size,
		const float learning_rate,
		__global float *loss,
		const __global float *x,
		const __global float *w1,
		const __global float *d_w1,
		__global float *r_w1,
		const __global float *b1,
		const __global float *d_b1,
		__global float *r_b1,
		const __global float *d_r_b1,
		__global float *r_a1,
		const __global float *d_r_a1,
		const __global float *w2,
		const __global float *d_w2,
		__global float *r_w2,
		const __global float *b2,
		const __global float *d_b2,
		__global float *r_b2,
		const __global float *d_r_b2,
		__global float *er,
		__global float *probs_cl,
		const __global int *labels
) {
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	
	*loss = 0;

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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if(global_id == 0 && local_id == 0) {
		*loss = ce_loss(
			labels,
			probs_cl,
			batchsize,
			classes
		) / (float) batchsize;
	}
}

void bp_ce_softmax(const int label, const __global float *in, __global float *res, const int w)
{
	for(int j=0; j < w; j++) {
		if(label == j) {
			res[j] = in[j] - 1;
		} else {
			res[j] = in[j];
		}
	}
}

void bp_w(const __global float *x, const __global float *prev, __global float *res, const int hw, const int h, const int w)
{
	//TODO - missing parallelization
	//TODO - some values are wrong
	
	mul_mat_transpA(x, prev, res, h, hw, w);
}

void bp_b(const __global float *prev, __global float *r, const int h, const int w)
{
    for(int i=0; i<w; i++){
        r[i] = 0;
        barrier(CLK_GLOBAL_MEM_FENCE);
        AtomicAdd(&r[i], prev[i]);
    }
}

void bp_x(const __global float *w_, const __global float *prev, __global float *res, const int hw, const int h, const int w)
{
	//TODO - missing parallelization
	//TODO - all values are wrong
	
	mul_mat_transpB(prev, w_, res, h, hw, w);
}

void bp_act(const __global float *x, const __global float *prev, __global float *res, const int w)
{
	for(int j=0; j < w; j++) {
		res[j] = (x[j] > 0) ? prev[j] : 0;
	}
}

__kernel void backprop(
		const int batchsize,
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int epochs,
		const int image_size,
		const float learning_rate,
		const __global float *loss,
		const __global float *x,
		const __global float *w1,
		__global float *d_w1,
		const __global float *r_w1,
		const __global float *b1,
		__global float *d_b1,
		const __global float *r_b1,
		__global float *d_r_b1,
		const __global float *r_a1,
		__global float *d_r_a1,
		const __global float *w2,
		__global float *d_w2,
		const __global float *r_w2,
		const __global float *b2,
		__global float *d_b2,
		const __global float *r_b2,
		__global float *d_r_b2,
		const __global float *er,
		const __global float *probs_cl,
		const __global int *labels
) {
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);

	bp_ce_softmax(
		labels[global_id],
		&probs_cl[global_id * classes],
		&d_r_b2[global_id * second_layer_neurons],
		second_layer_neurons
	);
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(global_id == 0 && local_id == 0) {
		bp_w(
			r_a1,
			d_r_b2,
			d_w2,
			batchsize,
			first_layer_neurons,
			second_layer_neurons
		);
	}
	bp_b(
		&d_r_b2[(global_id * second_layer_neurons)],
		d_b2,
		batchsize,
		second_layer_neurons
	);
	

	barrier(CLK_GLOBAL_MEM_FENCE);
	if(global_id == 0 && local_id == 0) {
		bp_x(
			w2,
			d_r_b2,
			d_r_a1,
			second_layer_neurons,
			batchsize,
			first_layer_neurons
		);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	bp_act(
		&r_b1[global_id * first_layer_neurons],
		&d_r_a1[global_id * first_layer_neurons],
		&d_r_b1[global_id * first_layer_neurons],
		first_layer_neurons
	);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(global_id == 0 && local_id == 0) {
		bp_w(
			x,
			d_r_b1,
			d_w1,
			batchsize,
			image_size * image_size,
			first_layer_neurons
		);
	}
	
	bp_b(
		&d_r_b1[global_id * first_layer_neurons],
		d_b1,
		batchsize,
		first_layer_neurons
	);
}

void update_param(__global float *param, const __global float *d_param, const float learning_rate, const int h, const int w)
{
	for(int i=0; i < h; i++) {
		for(int j=0; j < w; j++) {
			param[(i*w) + j] += (-learning_rate) * d_param[(i*w) + j];
		}
	}
}

__kernel void update_params(
		const int batchsize,
		const int classes,
		const int first_layer_neurons,
		const int second_layer_neurons,
		const int epochs,
		const int image_size,
		const float learning_rate,
		const __global float *loss,
		const __global float *x,
		__global float *w1,
		const __global float *d_w1,
		const __global float *r_w1,
		__global float *b1,
		const __global float *d_b1,
		const __global float *r_b1,
		const __global float *d_r_b1,
		const __global float *r_a1,
		const __global float *d_r_a1,
		__global float *w2,
		const __global float *d_w2,
		const __global float *r_w2,
		__global float *b2,
		const __global float *d_b2,
		const __global float *r_b2,
		const __global float *d_r_b2,
		const __global float *er,
		const __global float *probs_cl,
		const __global int *labels
) {
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);

	if(global_id == 0 && local_id == 0) {
		update_param(
			w1,
			d_w1,
			learning_rate,
			image_size*image_size,
			first_layer_neurons
		);
	}

	if(global_id == 1 && local_id == 0) {
		update_param(
			b1,
			d_b1,
			learning_rate,
			1,
			first_layer_neurons
		);
	}

	if(global_id == 2 && local_id == 0) {
		update_param(
			w2,
			d_w2,
			learning_rate,
			first_layer_neurons,
			second_layer_neurons
		);
	}

	if(global_id == 3 && local_id == 0) {
		update_param(
			b2,
			d_b2,
			learning_rate,
			1,
			second_layer_neurons
		);
	}
}
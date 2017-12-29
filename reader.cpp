#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <limits>

using namespace std;
const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int BATCHSIZE = 2;
const int IMAGE_WIDTH = 28;
const int CLASSES = 10;
const int FIRST_LAYER_NEURONS = 20;
const int SECOND_LAYER_NEURONS = CLASSES;


float** create_mat(int h, int w){
	float** ptr = new float*[h]; 		// pointer
	float* pool = new float[h*w];		// mempool
	for(int i=0;i<h;i++, pool += w){
		ptr[i] = pool;
	}
	return ptr;
}


void mul_mat(float** a, float** b, float** c, int h, int hw, int w){
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			float sum = 0;
			for(int k=0;k<hw;k++){
				sum += a[i][k] * b[k][j];
			}
			c[i][j] = sum;
		}
	}
}


void print_mat(float** mat, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			cout << mat[i][j] << " ";
		}
		cout << " " << endl;
	}
	cout << " " << endl;
}

struct image_label{
	unsigned char label;
	unsigned char pixel[IMAGE_WIDTH][IMAGE_WIDTH];
};

// loads mnist data into given image_label structs
void get_data(image_label* data, int data_size, string label_file, string image_file){
	char buff;

	ifstream l_file(label_file, ios::in|ios::binary);
	if(l_file.is_open()){
		l_file.seekg(8, ios::beg); // skip magic_number and size
		for(int i=0; i<data_size; i++){
			l_file.read(&buff, 1);
			data[i].label = static_cast<unsigned>(buff);
		}
		l_file.close();
	}
	else{
		cout << "File " << label_file << " not found."<< endl;
	}

	ifstream i_file(image_file, ios::in|ios::binary);
	if(i_file.is_open()){
		i_file.seekg(16, ios::beg); // skip magic_number/size/etc.
		for(int i=0; i<data_size; i++){
			for(int row=0; row<IMAGE_WIDTH; row++){
				for(int col=0; col<IMAGE_WIDTH; col++){
					i_file.read(&buff, 1);
					data[i].pixel[row][col] = static_cast<unsigned>(buff);
				}
			}
		}
		i_file.close();
	}
	else{
		cout << "File " << image_file << " not found."<< endl;
	}
}

// shitty console print test
void test_print(image_label* p){
	cout << +p->label << endl;
	for(int r=0; r<IMAGE_WIDTH; r++){
		for(int c=0; c<IMAGE_WIDTH; c++){
			cout << p->pixel[r][c] << " ";
		}
		cout << " " <<  endl;
	}
}

// loads image data into input matrix x
void load_input(float** x, image_label* il, int amount, int offset){
	for(int i=0; i<amount; i++){
		for(int j=0; j<IMAGE_WIDTH*IMAGE_WIDTH; j++){
			x[i][j] = il[i+offset].pixel[j/IMAGE_WIDTH][j%IMAGE_WIDTH];
		}
	}
}

// sets random values in mat from -0.1 to 0.1
void init_rand(float** mat, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			mat[i][j] = -0.1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.2)));
		}
	}
}

// adds single column bias b to whole matrix a - may also use matmul with adjusted dims/values
void broadcasted_add(float** a, float** b, float** c, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			c[i][j] = a[i][j] + b[0][j];
		}
	}
}


// applies a relu activation function on a, result in b
void relu_mat(float** a, float** b, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			b[i][j] = (a[i][j] > 0) ? a[i][j] : 0;
		}
	}
}

// applies exp function on a, result in b
void exp_mat(float** a, float** b, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			b[i][j] = exp(a[i][j]);
			if(isinf(b[i][j])){
				b[i][j] = numeric_limits<float>::max();
			}
		}
	}
}


// calculates softmax probabilities for scores in exponated mat a, result in b
void softmax_mat(float** a, float** b, int h, int w){
	for(int i=0; i<h; i++){
		float sum = 0;
		for(int j=0; j<w; j++){
			sum += a[i][j];
		}
		for(int j=0; j<w; j++){
			b[i][j] = a[i][j]/sum;
		}
	}
}

// computes cross_entropy data loss for probability mat a
float data_loss(image_label* il, float** a, int h, int w, int train_step){
	float sum = 0;
	float val;
	int il_index;		// current index of il according to row in mat a, h = BATCHSIZE
	for(int i=0; i<h; i++){
		il_index = i+train_step*h;
		val = -log(a[i][il[il_index].label]);
		if(a[i][il[il_index].label] == 0){
			val = -log(std::numeric_limits<float>::min());
		}
		sum += val;
	}
	return sum;
}

int main(int argc, char const *argv[]){
	srand (static_cast <unsigned> (time(0)));

	image_label* train_data = new image_label[TRAIN_SIZE]; 
	image_label* test_data = new image_label[TEST_SIZE];
	get_data(train_data, TRAIN_SIZE, "./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte");
	get_data(test_data, TEST_SIZE, "./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");

	// define mats for input/result/weight/biases and intermediary resuls
	float** x = create_mat(BATCHSIZE, IMAGE_WIDTH*IMAGE_WIDTH);

	float** w1 = create_mat(IMAGE_WIDTH*IMAGE_WIDTH, FIRST_LAYER_NEURONS);
	float** b1 = create_mat(1, FIRST_LAYER_NEURONS);
	float** rw1 = create_mat(BATCHSIZE, FIRST_LAYER_NEURONS);
	float** rb1 = create_mat(BATCHSIZE, FIRST_LAYER_NEURONS);

	float** rr1 = create_mat(BATCHSIZE, FIRST_LAYER_NEURONS);

	float** w2 = create_mat(FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS);
	float** b2 = create_mat(1, SECOND_LAYER_NEURONS);
	float** rw2 = create_mat(BATCHSIZE, SECOND_LAYER_NEURONS);
	float** rb2 = create_mat(BATCHSIZE, SECOND_LAYER_NEURONS);

	float** er = create_mat(BATCHSIZE, CLASSES);
	float** probs = create_mat(BATCHSIZE, CLASSES);

	// init biases and weights
	init_rand(w1, IMAGE_WIDTH*IMAGE_WIDTH, FIRST_LAYER_NEURONS);
	init_rand(b1, 1, FIRST_LAYER_NEURONS);
	init_rand(w2, FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS);
	init_rand(b2, 1, SECOND_LAYER_NEURONS);


	int train_step = 0;
	float loss = 0;
	// training loop
	load_input(x, train_data, BATCHSIZE, train_step*BATCHSIZE);

	mul_mat(x, w1, rw1, BATCHSIZE, IMAGE_WIDTH*IMAGE_WIDTH, FIRST_LAYER_NEURONS);
	broadcasted_add(rw1, b1, rb1, BATCHSIZE, FIRST_LAYER_NEURONS);
	relu_mat(rb1, rr1, BATCHSIZE, FIRST_LAYER_NEURONS);
	mul_mat(rr1, w2, rw2, BATCHSIZE, FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS);
	broadcasted_add(rw2, b2, rb2, BATCHSIZE, SECOND_LAYER_NEURONS);

	exp_mat(rb2, er, BATCHSIZE, CLASSES);
	softmax_mat(er, probs, BATCHSIZE, CLASSES);
	loss = data_loss(train_data, probs, BATCHSIZE, CLASSES, train_step) / (float)BATCHSIZE;


	// debugging
	test_print(&train_data[1]);
	test_print(&test_data[2]);
	print_mat(rb1, BATCHSIZE, FIRST_LAYER_NEURONS);
	print_mat(rb2, BATCHSIZE, SECOND_LAYER_NEURONS);
	print_mat(er, BATCHSIZE, CLASSES);
	print_mat(probs, BATCHSIZE, CLASSES);
	cout << "loss: " << loss << endl;

  	return 0;
}


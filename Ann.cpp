#include "Ann.h"


Ann::Ann(int train_size, int test_size, int batchsize, int classes, int first_layer_neurons, int second_layer_neurons,
         int epochs, Activation act) : train_size(train_size), test_size(test_size), batchsize(batchsize), classes(classes),
                       first_layer_neurons(first_layer_neurons), second_layer_neurons(second_layer_neurons),
                       epochs(epochs), act(act) {
}



float **Ann::create_mat(int h, int w) {
    float** ptr = new float*[h]; 		// pointer
    float* pool = new float[h*w];		// mempool
    for(int i=0;i<h;i++, pool += w){
        ptr[i] = pool;
    }
    return ptr;
}

void Ann::free_mat(float **toFree) {
    if(toFree == nullptr) {
        return;
    }
    delete[] toFree[0];
    delete[] toFree;
}

void Ann::mul_mat(float **a, float **b, float **c, int h, int hw, int w) {
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

void Ann::transp_mat(float **in, float **res, int h, int w) {
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            res[j][i] = in[i][j];
        }
    }
}

void Ann::load_input(float **x, imageLabel *il, int amount, int offset, int data_size) {
    for(int i=0; i<amount; i++){
        for(int j=0; j<IMAGE_SIZE*IMAGE_SIZE; j++){
            x[i][j] = il[(i+offset)%data_size].pixel[j/IMAGE_SIZE][j%IMAGE_SIZE];
            x[i][j] = (float)x[i][j]/255;  // map to 0-1
        }
    }
}

void Ann::init_rand(float **mat, int h, int w, float min, float max) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            mat[i][j] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2*max)));
        }
    }
}

void Ann::broadcasted_add(float **mat, float **bias, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            res[i][j] = mat[i][j] + bias[0][j];
        }
    }
}

void Ann::act_mat(float **in, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            switch(act){
                case Activation::RELU : res[i][j] = (in[i][j] > 0) ? in[i][j] : 0;  break;
                case Activation::LRELU : res[i][j] = (in[i][j] > 0) ? in[i][j] : (0.01 * in[i][j]); break;
                case Activation::TANH : res[i][j] = tanh(in[i][j]); break;
                case Activation::SIGMOID : res[i][j] = 1 / (1 + exp(- in[i][j])); break;
            }
        }
    }
}

void Ann::exp_mat(float **in, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            res[i][j] = exp(in[i][j]);
            if(isinf(res[i][j])){  // exp result to big - clamp to max result
                res[i][j] = std::numeric_limits<float>::max();
            }
        }
    }
}

void Ann::softmax_mat(float **in, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        float sum = 0;
        for(int j=0; j<w; j++){
            sum += in[i][j];
        }
        for(int j=0; j<w; j++){
            res[i][j] = in[i][j]/sum;
        }
    }
}

float Ann::ce_loss(imageLabel *il, float **a, int h, int w, int train_step, int data_size) {
    float sum = 0;
    float val;
    int il_index;		// current index of il according to row in mat a, h = BATCHSIZE
    for(int i=0; i<h; i++){
        il_index = (i+train_step*h)%data_size;
        val = -log(a[i][il[il_index].label]);
        if(a[i][il[il_index].label] == 0){  // log(0) undefined, choose lowest positive float instead
            val = -log(std::numeric_limits<float>::min());
        }
        sum += val;
    }
    return sum;
}

void Ann::bp_ce_softmax(imageLabel *il, float **in, float **res, int h, int w, int train_step, int data_size) {
    int il_index;
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            il_index = (i+train_step*h)%data_size;
            if(j == il[il_index].label){
                res[i][j] = in[i][j] - 1;
            }
            else{
                res[i][j] = in[i][j];
            }
        }
    }
}

void Ann::bp_w(float **x, float **prev, float **res, int hw, int h, int w) {
    float** xT = create_mat(h, hw);
    transp_mat(x, xT, hw, h);
    mul_mat(xT, prev, res, h, hw, w);

    free_mat(xT);
}

void Ann::bp_x(float **w_, float **prev, float **res, int hw, int h, int w) {
    float** wT = create_mat(hw, w);
    transp_mat(w_, wT, w, hw);
    mul_mat(prev, wT, res, h, hw, w);

    free_mat(wT);
}

void Ann::bp_b(float **prev, float **r, int h, int w) {
    float sum;
    for(int i=0; i<w; i++){
        sum = 0;
        for(int j=0; j<h; j++){
            sum += prev[j][i];
        }
        r[0][i] = sum;
    }
}

void Ann::bp_act(float **x, float **prev, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            switch(act){
                case Activation::RELU : res[i][j] = (x[i][j] > 0) ? prev[i][j] : 0;  break;
                case Activation::LRELU : res[i][j] = (x[i][j] > 0) ? prev[i][j] : (0.01*prev[i][j]); break;
                case Activation::TANH : res[i][j] = 1/(pow(cosh(x[i][j]), 2)) * prev[i][j]; break;
                case Activation::SIGMOID : res[i][j] = (1/(1+exp(-x[i][j]))) * (1-(1/(1+exp(-x[i][j])))); break;
            }
        }
    }
}

void Ann::update_param(float **param, float **d_param, float lr, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            param[i][j] += -lr*d_param[i][j];
        }
    }
}

void Ann::init_data(std::string train_label_file, std::string train_image_file, std::string test_label_file,
                    std::string test_image_file) {
    train_data = Reader::load_data(train_size, train_label_file, train_image_file);
    test_data = Reader::load_data(test_size, test_label_file, test_image_file);
}

void Ann::init_mats() {
    x = create_mat(batchsize, IMAGE_SIZE*IMAGE_SIZE);

    w1 = create_mat(IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons);
    b1 = create_mat(1, first_layer_neurons);
    r_w1 = create_mat(batchsize, first_layer_neurons);
    r_b1 = create_mat(batchsize, first_layer_neurons);
    r_a1 = create_mat(batchsize, first_layer_neurons);

    w2 = create_mat(first_layer_neurons, second_layer_neurons);
    b2 = create_mat(1, second_layer_neurons);
    r_w2 = create_mat(batchsize, second_layer_neurons);
    r_b2 = create_mat(batchsize, second_layer_neurons);

    er = create_mat(batchsize, classes);
    probs = create_mat(batchsize, classes);

    d_r_b2 = create_mat(batchsize, second_layer_neurons);
    d_w2 = create_mat(first_layer_neurons, second_layer_neurons);
    d_b2 = create_mat(1, second_layer_neurons);
    d_r_a1 = create_mat(batchsize, first_layer_neurons);
    d_r_b1 = create_mat(batchsize, first_layer_neurons);
    d_w1 = create_mat(IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons);
    d_b1 = create_mat(batchsize, first_layer_neurons);

    init_rand(w1, IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons, -0.1, 0.1);
    init_rand(b1, 1, first_layer_neurons, -0.1, 0.1);
    init_rand(w2, first_layer_neurons, second_layer_neurons, -0.1, 0.1);
    init_rand(b2, 1, second_layer_neurons, -0.1, 0.1);
}

float Ann::forward_pass(bool training, int step) {
    int data_size = training ? train_size : test_size;
    imageLabel* data = training ? train_data : test_data;

    load_input(x, data, batchsize, step * batchsize, data_size);

    mul_mat(x, w1, r_w1, batchsize, IMAGE_SIZE * IMAGE_SIZE, first_layer_neurons);
    broadcasted_add(r_w1, b1, r_b1, batchsize, first_layer_neurons);
    act_mat(r_b1, r_a1, batchsize, first_layer_neurons);
    mul_mat(r_a1, w2, r_w2, batchsize, first_layer_neurons, second_layer_neurons);
    broadcasted_add(r_w2, b2, r_b2, batchsize, second_layer_neurons);

    exp_mat(r_b2, er, batchsize, classes);
    softmax_mat(er, probs, batchsize, classes);
    float loss = ce_loss(data, probs, batchsize, classes, step, data_size) / (float) batchsize;
    return loss;
}

void Ann::backprop(bool training, int step) {
    int data_size = training ? train_size : test_size;
    imageLabel* data = training ? train_data : test_data;

    bp_ce_softmax(data, probs, d_r_b2, batchsize, second_layer_neurons, step, data_size);
    bp_w(r_a1, d_r_b2, d_w2, batchsize, first_layer_neurons, second_layer_neurons);
    bp_b(d_r_b2, d_b2, batchsize, second_layer_neurons);
    bp_x(w2, d_r_b2, d_r_a1, second_layer_neurons, batchsize, first_layer_neurons);
    bp_act(r_b1, d_r_a1, d_r_b1, batchsize, first_layer_neurons);
    bp_w(x, d_r_b1, d_w1, batchsize, IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons);
    bp_b(d_r_b1, d_b1, batchsize, first_layer_neurons);
}

void Ann::update_params(float learning_rate) {
    update_param(w1, d_w1, learning_rate, IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons);
    update_param(b1, d_b1, learning_rate, 1, first_layer_neurons);
    update_param(w2, d_w2, learning_rate, first_layer_neurons, second_layer_neurons);
    update_param(b2, d_b2, learning_rate, 1, second_layer_neurons);
}

float Ann::calc_acc(bool training, int step) {
    int data_size = training ? train_size : test_size;
    imageLabel* data = training ? train_data : test_data;
    int offset = step*batchsize;
    float acc;
    int correct = 0;
    for(int i=0; i<batchsize; i++){
        int max_index = 0;
        for(int j=0; j<10; j++){
            if(probs[i][j]>probs[i][max_index]){ // find class with highest probability
                max_index = j;
            }
        }
        if(data[(i+offset)%data_size].label==max_index){    // compare to correct class
            correct++;
        }
    }
    acc = (float)correct/batchsize;
    return acc;
}

Ann::~Ann() {
    free_mat(x);
    free_mat(w1);
    free_mat(b1);
    free_mat(r_w1);
    free_mat(r_b1);
    free_mat(r_a1);
    free_mat(w2);
    free_mat(b2);
    free_mat(r_w2);
    free_mat(r_b2);
    free_mat(er);
    free_mat(probs);
    free_mat(d_r_b2);
    free_mat(d_w2);
    free_mat(d_b2);
    free_mat(d_r_a1);
    free_mat(d_r_b1);
    free_mat(d_w1);
    free_mat(d_b1);

    delete[] test_data;
    delete[] train_data;
}





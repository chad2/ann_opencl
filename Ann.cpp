#include "Ann.h"


Ann::Ann(int train_size, int test_size, int batchsize, int classes, int first_layer_neurons, int second_layer_neurons,
         int epochs, Activation act) : train_size(train_size), test_size(test_size), batchsize(batchsize), classes(classes),
                       first_layer_neurons(first_layer_neurons), second_layer_neurons(second_layer_neurons),
                       epochs(epochs), act(act) {
}


/**
 * Creates a coherent pool of memory adressable in two dimensions.
 * @param h Height of created matrix.
 * @param w Width of created matrix.
 * @return Matrix with size h x w.
 */
float **Ann::create_mat(int h, int w) {
    float** ptr = new float*[h]; 		// pointer
    float* pool = new float[h*w];		// mempool
    for(int i=0;i<h;i++, pool += w){
        ptr[i] = pool;
    }
    return ptr;
}

/**
 * Frees memory of matrix created by create_mat.
 * @param toFree Matrix to be freed.
 */
void Ann::free_mat(float **toFree) {
    if(toFree == nullptr) {
        return;
    }
    delete[] toFree[0];
    delete[] toFree;
}

/**
 * Mutiplies two matrices.
 * @param a
 * @param b
 * @param c Result of a*b, with size h x w.
 * @param h Height of matrix a.
 * @param hw Width of matrix a, height of matrix b.
 * @param w Width of matrix b
 */
void Ann::mul_mat(float **a, float **b, float **c, int h, int hw, int w) {
    #if DISABLE_OPENCL == 1
        float** b_transp = create_mat(w, hw);
        transp_mat(b, b_transp, hw, w);

        for(int i=0; i < h; i++) {
            for(int j=0; j < w; j++) {
                c[i][j] = 0;
            }
        }

        for(int kk=0; kk<hw; kk+=this->BlockSize) {
            for(int jj=0; jj<w; jj+=this->BlockSize) {
                for(int i=0; i<h; i++) {
                    for(int j=jj; j< ((jj+this->BlockSize)> w ? w : (jj+this->BlockSize)); j++) {
                        float sum = 0;
                        for(int k=kk; k< ((kk+this->BlockSize) > hw ? hw : (kk+this->BlockSize)); k++) {
                            sum += a[i][k] * b_transp[j][k];
                        }
                        c[i][j] += sum;
                    }
                }
            }
        }
        free_mat(b_transp);
        b_transp = nullptr;
        
    #elif DISABLE_OPENCL == 0
        #define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))

        size_t TS = this->clm.getWorkGroupSize();
        const int hw_pad = CEIL_DIV(hw, TS) * TS;
        const int w_pad = CEIL_DIV(w, TS) * TS;
        const int h_pad = CEIL_DIV(h, TS) * TS;

        float **A_pad = create_mat(h_pad, hw_pad);
        for(int i=0; i < h_pad; i++) {
            for(int j=0; j < hw_pad; j++) {
                A_pad[i][j] = 0;
            }
        }
        for(int i=0; i < h; i++) {
            for(int j=0; j < hw; j++) {
                A_pad[i][j] = a[i][j];
            }
        }

        float **B_pad = create_mat(hw_pad, w_pad);
        for(int i=0; i < hw_pad; i++) {
            for(int j=0; j < w_pad; j++) {
                B_pad[i][j] = 0;
            }
        }
        for(int i=0; i < hw; i++) {
            for(int j=0; j < w; j++) {
                B_pad[i][j] = b[i][j];
            }
        }

        float **C_pad = create_mat(h_pad, w_pad);

        clm.cl_mul_mat(*B_pad, *A_pad, *C_pad, hw_pad, w_pad, h_pad);

        for(int i=0; i < h; i++) {
            for(int j=0; j < w; j++) {
                c[i][j] = C_pad[i][j];
            }
        }

        free_mat(A_pad);
        A_pad = nullptr;
        free_mat(B_pad);
        B_pad = nullptr;
        free_mat(C_pad);
        C_pad = nullptr;
    #endif

    #ifdef DEBUG
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            float sum = 0;
            for(int k=0;k<hw;k++){
                sum += a[i][k] * b[k][j];
            }
            if((std::max(sum, c[i][j]) - std::min(sum, c[i][j])) > 0.01) {
                std::cout << "i,j:" << i << "," << j << " " << sum << " != " << c[i][j] << std::endl;
                throw;
            }
        }
    }
    #endif
}
/**
 * Transposes matrix in.
 * @param in Matrix to be transposed.
 * @param res Result of transposed matrix in.
 * @param h Height of matrix in.
 * @param w Width of matrix in.
 */
void Ann::transp_mat(float **in, float **res, int h, int w) {
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            res[j][i] = in[i][j];
        }
    }
}

/**
 * Loads images into input matrix x, maps pixel values 0-1.
 * @param x Neural network input matrix.
 * @param il Dataset to be filled into matrix x. (test/train)
 * @param amount Amount of datapoints loaded into x. (amount <= batchsize)
 * @param offset Loaded date from il is offseted by offset.
 * @param data_size Size of dataset given in il.
 */
void Ann::load_input(float **x, imageLabel *il, int amount, int offset, int data_size) {
    for(int i=0; i<amount; i++){
        for(int j=0; j<IMAGE_SIZE*IMAGE_SIZE; j++){
            x[i][j] = il[(i+offset)%data_size].pixel[j/IMAGE_SIZE][j%IMAGE_SIZE];
            x[i][j] = (float)x[i][j]/255;  // map to 0-1
        }
    }
}

/**
 * Initialises matrix with uniform distribured values from min to max.
 * @param mat Matrix to be initialized with random values.
 * @param h Height of matrix mat.
 * @param w Width of matrix mat.
 * @param min Minimum of generated values.
 * @param max Maximum of generated values.
 */
void Ann::init_rand(float **mat, int h, int w, float min, float max) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            mat[i][j] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2*max)));
        }
    }
}

/**
 * Adds bias vector to each line in matrix.
 * @param mat Matrix sized h x w.
 * @param bias Biasvector with size 1 x w.
 * @param res Resulting matrix.
 * @param h Height of matrix mat.
 * @param w Width of matrix mat.
 */
void Ann::broadcasted_add(float **mat, float **bias, float **res, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            res[i][j] = mat[i][j] + bias[0][j];
        }
    }
}

/**
 * Applies an activiation function given matrix, to be used between network layers.
 * @param in Matrix to be passed through activation function.
 * @param res Result of activation(in).
 * @param h Height of matrix in.
 * @param w Width of matrix in.
 */
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

/**
 * Exponates values of given matrix elementwise.
 * @param in Matrix containing raw outputs from NN.
 * @param res Resulting matrix.
 * @param h Height of matrix in.
 * @param w Width of matrix in.
 */
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

/**
 * Applies softmax function on given matrix.
 * @param in Matrix containing exponated outputs from the NN.
 * @param res Matrix containing probabilities of classes for every datasample.
 * @param h Height of matrix in.
 * @param w Width of matrix in.
 */
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

/**
 * Computes cross entropy loss.
 * @param il imageLabel data used to compute probs in matrix a.
 * @param a Matrix filled with probabilities for datasamples.
 * @param h Height of matrix a.
 * @param w Width of matrix b.
 * @param train_step Current iterationstep.
 * @param data_size Size of datastrucure given in il.
 * @return Cross entropy loss.
 */
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
/**
 * Computes gradient for softmax layer.
 * @param il ImageLabel data used in forwardpass.
 * @param in Matrix containing probabilities.
 * @param res Gradient on raw score output from NN.
 * @param h Height of matrix in.
 * @param w Width of matrix in.
 * @param train_step Current iterationstep.
 * @param data_size Size of data given in il.
 */
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
/**
 * Computes gradient on weight matrix.
 * @param x Input of current layer.
 * @param prev Gradient from previous backprop layer, sized h x w.
 * @param res Resulting gradient on weight matrix w.
 * @param hw Height of matrix x, height of matrix prev.
 * @param h Width of matrix x.
 * @param w Width of matrix prev.
 */
void Ann::bp_w(float **x, float **prev, float **res, int hw, int h, int w) {
    float** xT = create_mat(h, hw);
    transp_mat(x, xT, hw, h);
    mul_mat(xT, prev, res, h, hw, w);

    free_mat(xT);
}

/**
 * Computes gradient on input matrix.
 * @param w Weights of current layer.
 * @param prev Gradient from previous backprop layer.
 * @param res Resulting gradient on input matrix x.
 * @param hw Width of prev, width of w_.
 * @param h Height of prev.
 * @param w Height of w_.
 */
void Ann::bp_x(float **w_, float **prev, float **res, int hw, int h, int w) {
    float** wT = create_mat(hw, w);
    transp_mat(w_, wT, w, hw);
    mul_mat(prev, wT, res, h, hw, w);

    free_mat(wT);
}
/**
 * Computes gradient on bias.
 * @param prev Gradient from previous backprop layer.
 * @param res Resulting gradient on bias matrix b.
 * @param h Height of matrix prev.
 * @param w Width of matrix prev, length of bias vector b.
 */
void Ann::bp_b(float **prev, float **res, int h, int w) {
    float sum;
    for(int i=0; i<w; i++){
        sum = 0;
        for(int j=0; j<h; j++){ 
            sum += prev[j][i];
        }
        res[0][i] = sum;
    }
}
/**
 * Computes gradient on activation function.
 * @param x Input to activation in forward pass.
 * @param prev Previous gradient.
 * @param res Resulting gradient.
 * @param h Height of matrix x.
 * @param w Width of matrix x.
 */
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
/**
 * Steps parameter into downward direction of errorfunction.
 * @param param Learnable parameter to be updated.
 * @param d_param Gradient on parameter param.
 * @param lr Learning rate.
 * @param h Height of param.
 * @param w Width of param.
 */
void Ann::update_param(float **param, float **d_param, float lr, int h, int w) {
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            param[i][j] += -lr*d_param[i][j];
        }
    }
}
/**
 * Loads mnist dataset into train_data and test_data fields.
 * @param train_label_file Path to training label file.
 * @param train_image_file Path to training image file.
 * @param test_label_file Path to test image file.
 * @param test_image_file Path to test image file.
 */
void Ann::init_data(std::string train_label_file, std::string train_image_file, std::string test_label_file,
                    std::string test_image_file) {
    train_data = Reader::load_data(train_size, train_label_file, train_image_file);
    test_data = Reader::load_data(test_size, test_label_file, test_image_file);
}
/**
 * Creates matrices for forward and backward step and initialises parameters with random values.
 */
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
/**
 * Computes a forward step for the entire network.
 * @param training Wether the network should use training or testdata.
 * @param step Current network step.
 * @return Cross entropy loss.
 */
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
/**
 * Computes gradients on all network paramters.
 * @param training Wether the network was using training or testdata.
 * @param step Current network step.
 */
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
/**
 * Performs a learning step on all network parameters.
 * @param learning_rate Learning rate.
 */
void Ann::update_params(float learning_rate) {
    update_param(w1, d_w1, learning_rate, IMAGE_SIZE*IMAGE_SIZE, first_layer_neurons);
    update_param(b1, d_b1, learning_rate, 1, first_layer_neurons);
    update_param(w2, d_w2, learning_rate, first_layer_neurons, second_layer_neurons);
    update_param(b2, d_b2, learning_rate, 1, second_layer_neurons);
}

/**
 *
 * @param training Wether the network was using training or testdata.
 * @param step Network step used, offset for imageLabel data.
 * @param visual Wether a visual sample of classifications should be displayed or not.
 * @return Classification accuracy on given step, precision depending on batchsize.
 */
float Ann::calc_acc(bool training, int step, bool visual) {
    int data_size = training ? train_size : test_size;
    imageLabel* data = training ? train_data : test_data;
    int offset = step*batchsize;
    float acc;
    int correct = 0;
    for(int i=0; i<batchsize; i++){
        int max_index = 0;
        for(int j=0; j<classes; j++){
            if(probs[i][j]>probs[i][max_index]){ // find class with highest probability
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
                std::cout << j << ": " << std::setprecision(3) << std::fixed << probs[i][j] << "  ";
            }
            imageLabel::print(&data[(i+offset)%data_size]);
            std::cout << std::endl;
        }
    }
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





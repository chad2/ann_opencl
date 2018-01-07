#ifndef UNTITLED3_ANN_H
#define UNTITLED3_ANN_H

#include "imageLabel.h"
#include "Reader.h"
#include "string"
#include <iostream>
#include <math.h>
#include <limits>

class Ann {
public:
    enum class Activation {RELU, LRELU, TANH, SIGMOID};

    Ann(int train_size, int test_size, int batchsize, int classes, int first_layer_neurons, int second_layer_neurons,
        int epochs, Activation act);

    virtual ~Ann();

    void init_data(std::string train_label_file, std::string train_image_file, std::string test_label_file,
                   std::string test_image_file);
    void init_mats();
    float forward_pass(bool training, int step);        // returns loss
    void backprop(bool training, int step);
    void update_params(float learning_rate);
    float calc_acc(bool training, int step);

private:
    int train_size;
    int test_size;
    int batchsize;
    int classes;
    int first_layer_neurons;
    int second_layer_neurons;
    int epochs;
    imageLabel* train_data;
    imageLabel* test_data;
    Activation act;

    float** x;      // input matrix

    float** w1;     // weights    -  first layer  -
    float** b1;     // bias                       -
    float** r_w1;   // result x*w1                -
    float** r_b1;   // result r_w1+b1             -
    float** r_a1;   // result activation of r_b1  -
    float** d_r_a1; //            -  first layer gradients  -
    float** d_r_b1; //                                      -
    float** d_w1;   //                                      -
    float** d_b1;   //                                      -

    float** w2;     // weights   -  second layer  -
    float** b2;     // bias                       -
    float** r_w2;   // result x                   -
    float** r_b2;   // result r_w1                -
    float** d_r_b2; //           -  second layer gradients  -
    float** d_w2;   //                                      -
    float** d_b2;   //                                      -

    float** er;     // exponated output from network for later use
    float** probs;  // resulting probabilities

    float** create_mat(int h, int w);
    void free_mat(float** toFree);
    void mul_mat(float** a, float** b, float** c, int h, int hw, int w);
    void transp_mat(float** in, float** res, int h, int w);
    void load_input(float** x, imageLabel* il, int amount, int offset, int data_size);
    void init_rand(float** mat, int h, int w, float min, float max);
    void broadcasted_add(float** mat, float** bias, float** res, int h, int w);
    void act_mat(float** in, float** res, int h, int w);
    void exp_mat(float** in, float** res, int h, int w);
    void softmax_mat(float** in, float** res, int h, int w);
    float ce_loss(imageLabel* il, float** a, int h, int w, int train_step, int data_size);

    void bp_ce_softmax(imageLabel* il, float** a, float** b, int h, int w, int train_step, int data_size);
    void bp_w(float** x, float** prev, float** res, int hw, int h, int w);
    void bp_x(float** w_, float** prev, float** res, int hw, int h, int w);
    void bp_b(float** prev, float** r, int h, int w);
    void bp_act(float** x, float** prev, float** res, int h, int w);
    void update_param(float** param, float** d_param, float lr, int h, int w);
};


#endif //UNTITLED3_ANN_H

#include <iostream>
#include <iomanip>
#include <time.h>
#include "Ann.h"


const int TRN_SIZE = 60000;
const int TST_SIZE = 10000;
const int BATCH_SIZE = 32;
const int CLASSES = 10;
const int FL_NEURONS = 20;
const int SL_NEURONS = CLASSES;
const int EPOCHS = 3;
const float LEARNING_RATE = 0.01;
const float DECAY_RATE = 0.0001;

int main() {
    //srand (static_cast <unsigned> (time(0)));
    srand (static_cast <unsigned> (0));
    Ann ann(TRN_SIZE, TST_SIZE, BATCH_SIZE, CLASSES, FL_NEURONS, SL_NEURONS, EPOCHS, Ann::Activation::RELU);
    ann.init_mats();
    ann.init_data("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte",
                   "./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");
    float learning_rate = LEARNING_RATE;
    std::cout << std::setprecision(4);
    float trn_loss, tst_acc;
    for(int step=0; step<(TRN_SIZE/BATCH_SIZE)*EPOCHS; step++){
        trn_loss = ann.forward_pass(true, step);
        ann.backprop(true, step);
        ann.update_params(learning_rate);
        if(step%(TRN_SIZE/BATCH_SIZE/5)==0){    // evaluate 5 times per epoch
            learning_rate *= (1.0 / (1.0 + DECAY_RATE * step));
            int rand_index = rand() % TST_SIZE;  // test on random sample of test_data
            ann.forward_pass(false, rand_index);
            tst_acc = ann.calc_acc(false, rand_index, false);
            std::cout << "step: " << std::setw(5) << step <<"/" << (TRN_SIZE/BATCH_SIZE)*EPOCHS;
            std::cout << " trn_loss: " << std::setw(9) << trn_loss;
            std::cout << " tst_acc: " << std::setw(4) << tst_acc;
            std::cout << " learning_rate: " << std::setw(9) << learning_rate << std::endl;
        }
    }
    ann.forward_pass(false, 0);
    ann.calc_acc(false, 0, true);
    return 0;
}
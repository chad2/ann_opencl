#include <iostream>
#include <time.h>
#include "Ann.h"


const int TRN_SIZE = 60000;
const int TST_SIZE = 10000;
const int BATCH_SIZE = 100;
const int CLASSES = 10;
const int FL_NEURONS = 50;
const int SL_NEURONS = CLASSES;
const int EPOCHS = 5;

int main() {
    srand (static_cast <unsigned> (time(0)));
    Ann* ann = new Ann(TRN_SIZE, TST_SIZE, BATCH_SIZE, CLASSES, FL_NEURONS, SL_NEURONS, EPOCHS, Ann::Activation::RELU);
    ann->init_mats();
    ann->init_data("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte",
                   "./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");
    float learning_rate = 0.005;
    float trn_loss, tst_acc;
    for(int step=0; step<(TRN_SIZE/BATCH_SIZE)*EPOCHS; step++){
        trn_loss = ann->forward_pass(true, step);
        ann->backprop(true, step);
        ann->update_params(learning_rate);
        if(step%100==0){
            int rand_index = rand() % TST_SIZE;  // test on random sample of test_data
            ann->forward_pass(false, rand_index);
            tst_acc = ann->calc_acc(false, rand_index);
            std::cout << "step: " << step << "/" << (TRN_SIZE/BATCH_SIZE)*EPOCHS;
            std::cout << "   trn_loss: " << trn_loss;
            std::cout << "   tst_acc: " << tst_acc << std::endl;
        }
    }
    return 0;
}
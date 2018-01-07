#ifndef UNTITLED3_READER_H
#define UNTITLED3_READER_H

#include "imageLabel.h"
#include <fstream>
#include <string>
#include <iostream>

class Reader {
public:
    static imageLabel* load_data(int data_size, std::string label_file, std::string image_file);
};


#endif //UNTITLED3_READER_H

#ifndef READER_H
#define READER_H

#include "imageLabel.h"
#include <fstream>
#include <string>
#include <iostream>

class Reader {
public:
    static imageLabel* load_data(int data_size, std::string label_file, std::string image_file);
};


#endif //READER_H

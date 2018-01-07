#include "imageLabel.h"
#include <iostream>

void imageLabel::print_imageLabel(imageLabel *il) {
    std::cout << +il->label << std::endl;
    for(int r=0; r<IMAGE_SIZE; r++){
        for(int c=0; c<IMAGE_SIZE; c++){
            std::cout << il->pixel[r][c] << " ";
        }
        std::cout << " " <<  std::endl;
    }
}

#include "imageLabel.h"

/**
 *
 * @param il
 */
void imageLabel::print(imageLabel *il) {
    std::cout << +il->label << std::endl;
    for(int r=0; r<IMAGE_SIZE; r++){
        for(int c=0; c<IMAGE_SIZE; c++){
        	il->pixel[r][c] > 10 ? std::cout << "O" : std::cout << " ";
        }
        std::cout << " " <<  std::endl;
    }
}

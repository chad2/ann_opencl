#include "Reader.h"
/**
 *
 * @param data_size
 * @param label_file
 * @param image_file
 * @return
 */
imageLabel *Reader::load_data(int data_size, std::string label_file, std::string image_file) {
    imageLabel* data = new imageLabel[data_size];
    char buff;

    std::ifstream l_file(label_file, std::ios::in|std::ios::binary);
    if(l_file.is_open()){
        l_file.seekg(8, std::ios::beg); // skip magic_number and size
        for(int i=0; i<data_size; i++){
            l_file.read(&buff, 1);
            data[i].label = static_cast<unsigned char>(buff);
        }
        l_file.close();
    }
    else{
        std::cout << "File " << label_file << " not found."<< std::endl;
    }

    std::ifstream i_file(image_file, std::ios::in|std::ios::binary);
    if(i_file.is_open()){
        i_file.seekg(16, std::ios::beg); // skip magic_number/size/etc.
        for(int i=0; i<data_size; i++){
            for(int row=0; row<IMAGE_SIZE; row++){
                for(int col=0; col<IMAGE_SIZE; col++){
                    i_file.read(&buff, 1);
                    data[i].pixel[row][col] = static_cast<unsigned char>(buff);
                }
            }
        }
        i_file.close();
    }
    else{
        std::cout << "File " << image_file << " not found."<< std::endl;
    }
    return data;
}

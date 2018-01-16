#ifndef IMAGELABEL_H
#define IMAGELABEL_H
#include <iostream>

const int IMAGE_SIZE = 28;

class imageLabel {
public:
    unsigned char label;
    unsigned char pixel[IMAGE_SIZE][IMAGE_SIZE];

	static void print(imageLabel* il);
};


#endif //IMAGELABEL_H
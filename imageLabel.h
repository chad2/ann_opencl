#ifndef UNTITLED3_IMAGELABEL_H
#define UNTITLED3_IMAGELABEL_H

const int IMAGE_SIZE = 28;

struct imageLabel {
    unsigned char label;
    unsigned char pixel[IMAGE_SIZE][IMAGE_SIZE];

    void print_imageLabel(imageLabel* il);
};


#endif //UNTITLED3_IMAGELABEL_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;

struct image_label{
	unsigned char label;
	unsigned char pixel[28][28];
};

// loads mnist data into given image_label structs
void get_data(image_label* data, int data_size, string label_file, string image_file){
	char buff;

	ifstream l_file(label_file, ios::in|ios::binary);
	if(l_file.is_open()){
		l_file.seekg(8, ios::beg); // skip magic_number and size
		for(int i=0; i<data_size; i++){
			l_file.read(&buff, 1);
			data[i].label = static_cast<unsigned>(buff);
		}
		l_file.close();
	}
	else{
		cout << "File " << label_file << " not found."<< endl;
	}

	ifstream i_file(image_file, ios::in|ios::binary);
	if(i_file.is_open()){
		i_file.seekg(16, ios::beg); // skip magic_number/size/etc.
		for(int i=0; i<data_size; i++){
			for(int row=0; row<28; row++){
				for(int col=0; col<28; col++){
					i_file.read(&buff, 1);
					data[i].pixel[row][col] = static_cast<unsigned>(buff);
				}
			}
		}
		i_file.close();
	}
	else{
		cout << "File " << image_file << " not found."<< endl;
	}
}

// shitty console print test
void test_print(image_label* p){
	cout << p->label << endl;
	for(int r=0; r<28; r++){
		for(int c=0; c<28; c++){
			cout << p->pixel[r][c] << " ";
		}
		cout << " " <<  endl;
	}
}

int main(int argc, char const *argv[]){
	image_label* train_data = new image_label[TRAIN_SIZE]; 
	image_label* test_data = new image_label[TEST_SIZE];
	get_data(train_data, TRAIN_SIZE, "./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte");
	get_data(test_data, TEST_SIZE, "./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");
	test_print(&train_data[1]);
	test_print(&test_data[2]);

  	return 0;
}


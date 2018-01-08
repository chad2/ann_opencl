PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OBJS = main.o Ann.o Reader.o imageLabel.o clMul.o

CPPFLAGS = -std=c++11 -Wall -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include"
CFLAGS = -Wall 
LFLAGS = -lm -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" -lOpenCL 

BUILD_MODE = run

ifeq ($(BUILD_MODE),debug)
	CFLAGS += -g -O0
	CFLAGS += -fsanitize=address
	LFLAGS += -lasan
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -O3 -mtune=native -march=native -funroll-loops -funroll-all-loops
	CFLAGS += -pie -fPIE
	LFLAGS += -pie -fPIE
else
	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif


all:	main

init:
	mkdir -p data
	wget -nc -P data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	gzip --name -d data/train-images-idx3-ubyte.gz
	wget -nc -P data http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	gzip --name -d data/train-labels-idx1-ubyte.gz
	wget -nc -P data http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	gzip --name -d data/t10k-images-idx3-ubyte.gz
	wget -nc -P data http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gzip --name -d data/t10k-labels-idx1-ubyte.gz
	

main:	$(OBJS)
	$(CXX) -o $@ $^ $(LFLAGS)

%.o:	$(PROJECT_ROOT)%.cpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o:	$(PROJECT_ROOT)%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -fr main $(OBJS)


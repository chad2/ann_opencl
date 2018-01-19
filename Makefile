PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OBJS = main.o Ann.o Reader.o imageLabel.o

ENABLE_FULL_OPENCL = 0
DISABLE_OPENCL = 0
ENABLE_SANITIZE = 0

BUILD_MODE = run

ifeq ($(BUILD_MODE),debug)
	CFLAGS = -g -O0
	LFLAGS =
	ifeq ($(ENABLE_SANITIZE),1)
		LFLAGS += -lasan
		CFLAGS += -fsanitize=address
	endif
	CFLAGS += -DDEBUG
else ifeq ($(BUILD_MODE),run)
	CFLAGS = -O3 -mtune=native -march=native -funroll-loops -funroll-all-loops
	CFLAGS += -pie -fPIE
	LFLAGS = -pie -fPIE
else
	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

CPPFLAGS += -std=c++11 -Wall
CFLAGS += -Wall -DENABLE_FULL_OPENCL=$(ENABLE_FULL_OPENCL) -DDISABLE_OPENCL=$(DISABLE_OPENCL)
LFLAGS += -lm

ifeq ($(DISABLE_OPENCL),0)
	OBJS += clMul.o AnnOpenCL.o
	ifeq ($(OS),Windows_NT)
		CPPFLAGS += -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include"
		LFLAGS += -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" -lOpenCL
	else
		LFLAGS += -lOpenCL
	endif
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


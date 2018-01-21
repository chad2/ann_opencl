# ann_opencl

# how to use
1. `make init`
    - creates 'data' folder
    - downloads dataset for training and testing
2. `make`
    - compiles the project 
3. (optional) `make BUILD_MODE=x`
    - by default 'x' is set to 'run'
    - other option is 'debug'

## Compilation options
- `ENABLE_FULL_OPENCL=1`: the full ANN will run in OpenCL
- `DISABLE_OPENCL=1`: disables everything related to OpenCL and the program runs on CPU
- `ENABLE_SANITIZE=1`: enables `-fstanitize=address` and `-lasan` to find memory errors (uses a lot of RAM\!)

## runtime 
### i5 3570k, GTX 1060 (6 GB) , FL_NEURONS = 256, BATCHSIZE = 128, EPOCHS = 5
* `./main` :          23.5 seconds
* `python3 ./mnist_keras.py` :  12.79 seconds


## References
[LeCun et al., 1998a]
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
    
[Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/)

[CNugteren / myGEMM - OpenCL Matrix multiplication](https://github.com/CNugteren/myGEMM)
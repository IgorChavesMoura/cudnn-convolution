# Basic Convolution using CUDA cuDNN 
Basic convolution applying edge detection kernel to some samples using CUDA C++ and cuDNN 

* Project made on Linux 
* If you are new to CUDA, check https://github.com/IgorChavesMoura/cuda-basic
* How to install cuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
* How to compile: nvcc -o convolution convolution.cu -lcudnn -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

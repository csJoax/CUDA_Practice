cmake_minimum_required(VERSION 2.8)
project(CUDA_Practice)
find_package(CUDA REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)

INCLUDE_DIRECTORIES(
    ./include/
)

CUDA_ADD_EXECUTABLE(basis 
    basis.cu
)

CUDA_ADD_EXECUTABLE(conv 
    conv.cu
)

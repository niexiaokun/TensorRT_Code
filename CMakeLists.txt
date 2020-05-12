cmake_minimum_required(VERSION 3.0)
project(TRT_Demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

# setup CUDA
find_package(CUDA)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_50,code=sm_50  #gtx 750ti
        -gencode arch=compute_53,code=sm_53  #tegra tx1
        -gencode arch=compute_61,code=sm_61  #gtx 1060
        -gencode arch=compute_62,code=sm_62  #tegra tx2
)

if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")
    set (
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -gencode arch=compute_72,code=sm_72 #xavier
            -gencode arch=compute_75,code=sm_75 #rtx2080
    )
endif()

find_package(OpenCV REQUIRED)

#set(TRT_DIR /home/kun/TensorRT-5.1.2.2)
set(TRT_DIR /home/kun/TensorRT-5.1.5.0)

include_directories(common)
include_directories(${TRT_DIR}/include)
include_directories(/usr/local/cuda/include)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/cuda/lib64/stubs)
link_directories(${TRT_DIR}/lib)
link_directories(${TRT_DIR}/lib/stubs)

add_executable(trt_ssd ssd_main.cpp common/logger.cpp)

target_link_libraries(trt_ssd ${TRT_DIR}/lib/stubs/libnvinfer_plugin.so)
target_link_libraries(trt_ssd ${TRT_DIR}/lib/stubs/libnvinfer.so)
target_link_libraries(trt_ssd ${TRT_DIR}/lib/stubs/libnvonnxparser.so)
target_link_libraries(trt_ssd ${TRT_DIR}/lib/stubs/libnvparsers.so)
#target_link_libraries(trt_ssd /usr/local/cuda/lib64/stubs/libcuda.so)
#target_link_libraries(trt_ssd /usr/local/cuda/lib64/stubs/libcublas.so)
#target_link_libraries(trt_ssd /usr/local/cuda/lib64/stubs/libcurand.so)
target_link_libraries(trt_ssd /usr/local/cuda/lib64/libcudart.so)
target_link_libraries(trt_ssd ${OpenCV_LIBS})
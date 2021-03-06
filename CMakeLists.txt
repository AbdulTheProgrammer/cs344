############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# Upgraded by Tony Knight for local install with GTX1060 Win 10 CUDA 8 VS2015
# <email> tntknight@gmail.com </email>

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cs344)

find_package(OpenCV 3.2 REQUIRED)
find_package(CUDA 8.0 REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect
  # whether to build x64.  We tell it to support most devices, though,
  # to make sure more people can easily run class code without knowing
  # about this compiler argument

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  #  -gencode arch=compute_61,code=sm_61)

  #  -gencode arch=compute_30,code=sm_30;
  #  -gencode arch=compute_35,code=sm_35;
  #  -gencode arch=compute_50,code=sm_50;
    -gencode arch=compute_52,code=sm_52)
  #  -gencode arch=compute_60,code=sm_60;



  # need to set fused multiply and add to off to get same results as reference. see
  # https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fmad=false)


  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory ("Problem Sets/Problem Set 1")
  add_subdirectory ("Problem Sets/Problem Set 2")
  add_subdirectory ("Problem Sets/Problem Set 3")
  add_subdirectory ("Problem Sets/Problem Set 4")
  add_subdirectory ("Problem Sets/Problem Set 5")
  add_subdirectory ("Problem Sets/Problem Set 6")

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

cmake_minimum_required(VERSION 3.23)

project(brown CXX CUDA)

set (CMAKE_CXX_STANDARD 17)


include(FetchContent)
FetchContent_Declare(
  rd123
  #SOURCE_DIR /home/shihab/src/rnd
  GIT_REPOSITORY https://github.com/DEShawResearch/random123.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(rd123)

# HIP Stuff
set(ROOT_PATH /mnt/ufs18/home-158/khanmd)
set(ROCM_PATH ${ROOT_PATH}/hippy/clr/build/install)
set(RND_PATH ${ROOT_PATH}/rnd/include)


set(HOST_COMPILER g++)
set(CMAKE_CXX_COMPILER ${ROOT_PATH}/hippy/HIPCC/bin/hipcc)


set(GENCODE_FLAGS -arch=sm_70)
set(SRCS brownian.cpp)

# Include directories
include_directories(${ROCM_PATH}/include)

# Link directories
link_directories(${ROCM_PATH}/lib)

add_executable(brown ${SRCS})

target_include_directories(brown PRIVATE ${rd123_SOURCE_DIR}/include)

# Compile options for brown target
set_target_properties(brown PROPERTIES CUDA_STANDARD 17) 
target_compile_options(brown PRIVATE ${NVCCFLAGS} ${GENCODE_FLAGS})

# Link libraries
target_link_libraries(brown)
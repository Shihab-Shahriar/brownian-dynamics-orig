
cmake_minimum_required(VERSION 3.23)

project(brown CXX CUDA)

set (CMAKE_CXX_STANDARD 17)

# Kokkos Stuff
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in Kokkos")
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "lambda")
set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Ampere80")
#set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "volta 70")
#set(Kokkos_ARCH_TURING75 ON CACHE BOOL "turing 75")

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        master
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)

# HIP Stuff
set(ROOT_PATH /mnt/ufs18/home-158/khanmd)
set(ROCM_PATH ${ROOT_PATH}/hippy/clr/build/install)
set(RND_PATH ${ROOT_PATH}/rnd/include)


set(HOST_COMPILER g++)
set(NVCC ${ROOT_PATH}/hippy/HIPCC/bin/hipcc)

set(GENCODE_FLAGS -arch=sm_80)
set(SRCS brownian.cpp)

# Include directories
include_directories(${ROCM_PATH}/include)

# Link directories
link_directories(${ROCM_PATH}/lib)

add_executable(brown ${SRCS})

target_link_libraries(brown Kokkos::kokkos)
set_property(TARGET brown PROPERTY CUDA_ARCHITECTURES 80)

# Compile options for brown target
set_target_properties(brown PROPERTIES CUDA_STANDARD 17) 
target_compile_options(brown PRIVATE ${NVCCFLAGS} ${GENCODE_FLAGS})

# Link libraries
target_link_libraries(brown)
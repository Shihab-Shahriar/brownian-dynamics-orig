# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.23)

# Project's name
project(brown CXX CUDA)

set(ARCH "sm_80" CACHE STRING "GPU architecture to compile for")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH} -Xcompiler -Wall,-Wextra,-Werror")

include(FetchContent)

FetchContent_Declare(
  crng
  # SOURCE_DIR /mnt/ufs18/home-158/khanmd/rnd
  GIT_REPOSITORY https://github.com/Shihab-Shahriar/rnd.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(crng)

# Later in the file, when defining their targets:
add_executable(brown brownian.cu)
target_link_libraries(brown PRIVATE crng)
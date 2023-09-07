This is a simple 2D brownian dynamics simulation. Particles are moved by drag force (dependent on velocity) and random force. This has been used to compare the performance of different random number libraries in paper [1].

The main program is written in [HIP C++](https://github.com/ROCm-Developer-Tools/HIP). You need to have HIP  installed on your system to compile and run the program. Refer to [this link](https://github.com/ROCm-Developer-Tools/HIP/blob/develop/docs/developer_guide/build.md) for instructions on how to install HIP from source.

These programs has been tested on GCC 12.2.0, CUDA 12.0.0, HIP 5.6.x and CMake 3.24.3.

## Usage
Each library sits in its own branch. To compile and run the simulation, please refer to the README.md in each branch. In all cases, append the HIP's library path to `LD_LIBRARY_PATH` environment variable before compiling. For example, if you have installed HIP in `/opt/rocm/hip`, then you should run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib` before running `make`.

Branch name | Library
------------|--------
`cuda`    | `curand`
`rng`   | `our library(?)`
`rocm`     | `rocRAND`
`kokkos`  | `kokkos::Random_XorShift64_Pool`
`r123`  | `Random123`

[1] https://arxiv.org/abs/XXX.XXXX


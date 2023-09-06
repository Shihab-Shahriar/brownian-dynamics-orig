#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#define FUNCTION_MACRO __host__ __device__
#define PI           3.14159265358979323846 
#define SCALAR      double

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkHipErrors(val) check_hip( (val), #val, __FILE__, __LINE__ )

void check_hip(hipError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "HIP error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func <<" "<<hipGetErrorString(result)<< "' \n";
        // Make sure we call HIP Device Reset before exiting
        hipDeviceReset();
        exit(99);
    }
}

using ExecutionSpace = Kokkos::Cuda;
using CudaMemorySpace = ExecutionSpace::memory_space;

const double RADIUS = 1.0;
const int N = 1000000;
const double dt = 0.05;
const double T = 64.0;
const double GAMMA = 1.0;
const double mass = 1.0;
const int STEPS = 10000;

const int windowWidth = 800;
const int windowHeight = 600;

struct Particle {
    double x = 0;
    double y = 0;
    double vx = 0;
    double vy = 0;

    int pid = 0;

    FUNCTION_MACRO Particle(double x, double y) : x(x), y(y) {}


    FUNCTION_MACRO void update(double dx, double dy) {
        x += dx; 
        if(x < 0)
            x = 0;
        else if(x > windowWidth)
            x = windowWidth;

        y += dy;
        if(y < 0)
            y = 0;
        else if(y > windowHeight)
            y = windowHeight;
    }
};

// __global__ void rand_init(RNG *rand_state) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if(i >= N) 
//         return;

//     // TODO: Each thread gets different seed, same sequence for
//     // performance improvement of about 2x!
//     curand_init(1984, i, 0, &rand_state[i]);
// }

__global__ void init_particles(Particle *particles, Kokkos::Random_XorShift64_Pool<CudaMemorySpace> random_pool) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i >= N) return;

    Particle p = particles[i];
    p.pid = i;

    Kokkos::Random_XorShift64<CudaMemorySpace> gen =  random_pool.get_state();

    auto x = gen.drand() * double(windowWidth) - 1.0;
    auto y = gen.drand() * double(windowHeight) - 1.0;
    p.update(x, y);

    p.vx = gen.drand() * 100. - 50.0;
    p.vy = gen.drand() * 100. - 50.0;

    random_pool.free_state(gen);
    particles[i] = p;
}

__global__ void apply_forces(Particle *particles, Kokkos::Random_XorShift64_Pool<CudaMemorySpace> random_pool,
         double sqrt_dt, int counter) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i >= N) return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Apply random force
    Kokkos::Random_XorShift64<CudaMemorySpace> gen =  random_pool.get_state();
    
    p.vx += (gen.drand() * 2.0 - 1.0) * sqrt_dt;
    p.vy += (gen.drand() * 2.0 - 1.0) * sqrt_dt;
    
    random_pool.free_state(gen);
    particles[i] = p;
}

__global__ void update_positions(Particle *particles) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i >= N) return;

    Particle p = particles[i];

    // Check for collisions with box boundaries
    if (p.x - RADIUS < 0 || p.x + RADIUS > windowWidth) {
        p.vx *= -1;
    }
    if (p.y - RADIUS < 0 || p.y + RADIUS > windowHeight) {
        p.vy *= -1;
    }
    // Update positions
    p.update(p.vx * dt, p.vy * dt);

    particles[i] = p;
}

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force
    std::cout << "sqrt_dt: " << sqrt_dt << "\n";

    const double density = (N * PI * RADIUS* RADIUS) / (windowWidth * windowHeight);
    std::cout << "density: " << density << "\n";

    Kokkos::Random_XorShift64_Pool<CudaMemorySpace> random_pool(12345);

    Particle *particles;
    checkHipErrors(hipMallocManaged((void **)&particles, N * sizeof(Particle)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;


    hipLaunchKernelGGL(init_particles, dim3(nblocks), dim3(nthreads), 0, 0, particles, random_pool);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize());

    int iter = 0;
    while (iter++ < STEPS) {
        hipLaunchKernelGGL(apply_forces, dim3(nblocks), dim3(nthreads), 0, 0, particles, random_pool, sqrt_dt, iter);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());

        hipLaunchKernelGGL(update_positions, dim3(nblocks), dim3(nthreads), 0, 0, particles);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());
    }
}

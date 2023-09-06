#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>
#include <phillox.h>


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

typedef Phillox RNG; // HIP random number generator type

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


__global__ void init_particles(Particle *particles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    p.pid = i;

    RNG local_rand_state(p.pid, 0);
    auto x = local_rand_state.rand<double>() * double(windowWidth) - 1.0;
    auto y = local_rand_state.rand<double>() * double(windowHeight) - 1.0;
    p.update(x, y);

    p.vx = local_rand_state.rand<double>() * 100. - 50.0;
    p.vy = local_rand_state.rand<double>() * 100. - 50.0;

    particles[i] = p;
}

__global__ void apply_forces(Particle *particles, double sqrt_dt, int counter){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Apply random force
    RNG local_rand_state(p.pid, counter);
    p.vx += (local_rand_state.rand<double>()  * 2.0 - 1.0) * sqrt_dt;
    p.vy += (local_rand_state.rand<double>()  * 2.0 - 1.0) * sqrt_dt;
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

int main() {
    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force
    std::cout << "sqrt_dt: " << sqrt_dt << "\n";

    const double density = (N * PI * RADIUS* RADIUS) / (windowWidth * windowHeight);
    std::cout << "density: " << density << "\n";


    Particle *particles;
    checkHipErrors(hipMallocManaged((void **)&particles, N * sizeof(Particle)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;


    hipLaunchKernelGGL(init_particles, dim3(nblocks), dim3(nthreads), 0, 0, particles);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize());

    int iter = 0;
    while (iter++ < STEPS) {
        hipLaunchKernelGGL(apply_forces, dim3(nblocks), dim3(nthreads), 0, 0, particles, sqrt_dt, iter);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());

        hipLaunchKernelGGL(update_positions, dim3(nblocks), dim3(nthreads), 0, 0, particles);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());
    }
}

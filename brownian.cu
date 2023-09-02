#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

#include <phillox.h>

#define DEVICE __host__ __device__

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func <<" "<<cudaGetErrorString(result)<< "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

typedef Phillox RNG;


// Radius of particles
const double RADIUS = 2.0;
const int N = 1000000; // Number of particles

// Parameters for Lennard-Jones potential
const double epsilon = 1.0; // Depth of potential well
const double sigma = 2 * RADIUS;  // Finite distance at which the inter-particle potential is zero
const double cutoff_distance = 2.5 * sigma;

const double dt = 0.01; // Time step
const double T = 1.0; // Temperature
const double GAMMA = 1.0; // Drag coefficient
const double mass = 1.0; // Mass of particles
const int steps = 10000; // Number of simulation steps

//Sim Box parameters
const int windowWidth = 800;
const int windowHeight = 600;
const int cell_max_x = ((double)windowWidth + cutoff_distance - 1.0f) / cutoff_distance;
const int cell_max_y = ((double)windowHeight + cutoff_distance - 1.0f) / cutoff_distance;


struct Particle {
    double x = 0;
    double y = 0;
    double vx = 0;
    double vy = 0;

    int cell_id;

    DEVICE Particle(){

    };

    DEVICE Particle(float x, float y) : x(x), y(y) 
    {

    }

    DEVICE void update(float dx, float dy) {
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


__global__ void init_particles(Particle *particles, int counter){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    RNG rand_state(i, counter);
    Particle p = particles[i];
    auto x = rand_state.rand() * float(windowWidth) - 1.0f;
    auto y = rand_state.rand() * float(windowHeight) - 1.0f;
    p.update(x, y);
    particles[i] = p;
}

__global__ void assign_cells(Particle *particles ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    int cell_x = particles[i].x / cutoff_distance;
    int cell_y = particles[i].y / cutoff_distance;

    particles[i].cell_id = cell_x * cell_max_y + cell_y;
}

struct ParticleComparator{
    DEVICE bool operator()(const Particle& p1, const Particle& p2) const {
        return p1.cell_id < p2.cell_id;
    }
};

void rebuild_cell_list(Particle* particles, int *cell_list_idx, int nblocks, int nthreads){
    assign_cells<<<nblocks, nthreads>>>(particles);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<Particle> dev_ptr(particles);
    thrust::sort(dev_ptr, dev_ptr + N, ParticleComparator());

    // Find cell boundaries
    cell_list_idx[0] = 0;
    int max_cell_id = cell_max_x * cell_max_y - 1;
    int i = 0; // index of particles, 0<i<N;
    for(int cell_id = 0; cell_id <= max_cell_id; cell_id++){
        while(i < N && particles[i].cell_id == cell_id){
            i++;
        }
        cell_list_idx[cell_id + 1] = i;
    } 

}

// Compute collision forces based on Lennard-Jones potential
DEVICE void compute_collision_force(Particle& p1, Particle& p2, double& fx, double& fy) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx * dx + dy * dy;

    double r = std::sqrt(r2);
    if(r>= 3* sigma)
        return;
    
    // Lennard-Jones force magnitude
    double r_inv = sigma / r;
    double r_inv6 = r_inv * r_inv * r_inv * r_inv * r_inv * r_inv;
    double r_inv12 = r_inv6 * r_inv6;
    double f_magnitude = 24.0 * epsilon * (2*r_inv12 - r_inv6) / r;
    //printf("r: %f, f_magnitude: %f\n", r, f_magnitude);

    fx = f_magnitude * (dx / r);
    fy = f_magnitude * (dy / r);
}

__global__ void collision(Particle *particles, int* cell_list_idx){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    int cell_id = particles[i].cell_id;

    for(int di = -1; di<=1; di++){
        for(int dj = -1; dj<=1; dj++){
            int neighbor_cell_id = cell_id + di * cell_max_y + dj;
            if(neighbor_cell_id < 0 || neighbor_cell_id >= cell_max_x * cell_max_y)
                continue;
            int start_idx = cell_list_idx[neighbor_cell_id];
            int end_idx = cell_list_idx[neighbor_cell_id + 1];
            for(int j = start_idx; j < end_idx; j++){
                if(i == j)
                    continue;
                double fx, fy;
                compute_collision_force(particles[i], particles[j], fx, fy);
                particles[i].vx += fx * dt / mass;
                particles[i].vy += fy * dt / mass;
            }
        }
    }
}


__global__ void apply_forces(Particle *particles, int counter, double sqrt_dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Apply random force
    RNG rng(i, counter);
    p.vx += rng.randn(0.0, sqrt_dt);
    p.vy += rng.randn(0.0, sqrt_dt);
    particles[i] = p;
}

__global__ void update_positions(Particle *particles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
        
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



int main(){
    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force


    // allocate particles
    Particle *particles;
    checkCudaErrors(cudaMallocManaged((void **)&particles, N * sizeof(Particle)));

    // Allocate cell list indexes
    int *cell_list_idx;
    checkCudaErrors(cudaMallocManaged((void **)&cell_list_idx, (N+1) * sizeof(int)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;

    // Initialize particles
    init_particles<<<nblocks, nthreads>>>(particles, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Simulation loop
    int iter = 0;
    while(iter < steps){
        if(iter % 9 == 0){ //hoombd-blue does it every 9th step
            rebuild_cell_list(particles, cell_list_idx, nblocks, nthreads);
        }

        // Compute forces
        collision<<<nblocks, nthreads>>>(particles, cell_list_idx);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        apply_forces<<<nblocks, nthreads>>>(particles, iter+1, sqrt_dt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        
        update_positions<<<nblocks, nthreads>>>(particles);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        iter++;
    }

}




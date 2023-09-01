#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <vector>

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

typedef curandStatePhilox4_32_10_t RNG;


// Radius of particles
const double RADIUS = 2.0;
const int N = 100; // Number of particles

// Parameters for Lennard-Jones potential
const double epsilon = 1.0; // Depth of potential well
const double sigma = 2 * RADIUS;  // Finite distance at which the inter-particle potential is zero
const double cutoff_distance = 2.5 * sigma;

const double dt = 0.01; // Time step
const double T = 1.0; // Temperature
const double GAMMA = 1.0; // Drag coefficient
const double mass = 1.0; // Mass of particles

//Sim Box parameters
const int windowWidth = 800;
const int windowHeight = 600;
const int cell_max_x = ((double)windowWidth + cutoff_distance - 1.0f) / cutoff_distance;
const int cell_max_y = ((double)windowHeight + cutoff_distance - 1.0f) / cutoff_distance;

const sf::Color colors[4] = {sf::Color::Green, sf::Color::Red, sf::Color::Blue, sf::Color::Yellow};

struct Particle {
    double x = 0;
    double y = 0;
    double vx = 0;
    double vy = 0;
    int col = 0;

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

__global__ void rand_init(RNG *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) 
        return;

    // TODO: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984, i, 0, &rand_state[i]);
}

__global__ void init_particles(Particle *particles, RNG * rand_state){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    p.col = i % 4;
    RNG local_rand_state = rand_state[i];
    auto x = curand_uniform(&local_rand_state) * float(windowWidth) - 1.0f;
    auto y = curand_uniform(&local_rand_state) * float(windowHeight) - 1.0f;
    p.update(x, y);
    rand_state[i] = local_rand_state;
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

template<typename RNG>
DEVICE double get_randn(RNG* rand_state, double mean, double std_dev){
    double res = curand_normal(rand_state) * std_dev + mean;
    return res;
}

template <typename RNG>
__global__ void apply_forces(Particle *particles, RNG* rand_state, double sqrt_dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Apply random force
    RNG local_rand_state = rand_state[i];
    p.vx += get_randn(&local_rand_state, 0.0, sqrt_dt);
    p.vy += get_randn(&local_rand_state, 0.0, sqrt_dt);
    rand_state[i] = local_rand_state;
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

std::tuple<double, double, double, double>
find_extreme_positions(Particle* particles){
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();;
    for(int i=0; i<N; i++){
        Particle p = particles[i];
        if(p.x < min_x)
            min_x = p.x;
        if(p.x > max_x)
            max_x = p.x;
        if(p.y < min_y)
            min_y = p.y;
        if(p.y > max_y)
            max_y = p.y;
    }
    return std::make_tuple(min_x, max_x, min_y, max_y);
}

std::tuple<double, double>
find_velocity_magn(Particle* particles){
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    for(int i=0; i<N; i++){
        Particle p = particles[i];
        double v = std::sqrt(p.vx * p.vx + p.vy * p.vy);
        if(v < min_v)
            min_v = v;
        if(v > max_v)
            max_v = v;
    }
    return std::make_tuple(min_v, max_v);
}

void get_all_velocity_magnitudes(Particle* particles, std::vector<double>& vels){
    for(int i=0; i<N; i++){
        Particle p = particles[i];
        double v = std::sqrt(p.vx * p.vx + p.vy * p.vy);
        vels[i] = v;
    }
}


int main(){
    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force
    std::cout<< "sqrt_dt: "<<sqrt_dt<<std::endl;

    // Random number generator setup
    RNG *d_rand_states;
    checkCudaErrors(cudaMalloc((void **)&d_rand_states, N*sizeof(RNG)));


    // allocate particles
    Particle *particles;
    checkCudaErrors(cudaMallocManaged((void **)&particles, N * sizeof(Particle)));

    // Allocate cell list indexes
    int *cell_list_idx;
    checkCudaErrors(cudaMallocManaged((void **)&cell_list_idx, (N+1) * sizeof(int)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    rand_init<<<nblocks, nthreads>>>(d_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize particles
    init_particles<<<nblocks, nthreads>>>(particles, d_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // output initial positions
    // std::cout << "Initial positions:\n";
    // for (int i=0; i<N; i++) {
    //     std::cout << "Particle " << i << ": " << particles[i].x << ", " << particles[i].y << "\n";
    // }

    // Set up SFML window
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Brownian Dynamics Simulation");
    window.setFramerateLimit(120);

    sf::Font font;
    if (!font.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf")) {
        // handle error
    }

    sf::Text fpsText;
    fpsText.setFont(font);
    fpsText.setCharacterSize(24); // in pixels
    fpsText.setFillColor(sf::Color::White);
    fpsText.setPosition(10.f, 10.f);

    sf::Clock clock;

    // Array of shapes for each particle
    sf::CircleShape shapes[N];
    for (int i=0; i<N; i++) {
        shapes[i].setRadius(RADIUS);
        shapes[i].setPosition(particles[i].x, particles[i].y);
    }

    bool isRunning = true;
    std::vector<double> vels_prev(N);
    std::vector<double> vels_curr(N);

    // Simulation loop
    int iter = 0;
    while (window.isOpen()) {
        iter++;
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Q) {
                    isRunning = false;
                    std::cout << "Simulation Paused. Press 'R' to resume." << std::endl;

                    auto [min_x, max_x, min_y, max_y] = find_extreme_positions(particles);
                    std::cout << min_x<<", "<<max_x<<", "<<min_y<<", "<<max_y<<std::endl;

                    auto [min_v, max_v] = find_velocity_magn(particles);
                    std::cout << min_v<<", "<<max_v<<std::endl;

                    vels_prev = vels_curr;
                    get_all_velocity_magnitudes(particles, vels_curr);
                    //print all force magnitudes in last time step
                    for(int i=0; i<N; i++){
                        double force = (vels_curr[i] - vels_prev[i])  / dt;
                        std::cout<<force<<", ";
                    }
                    std::cout<<std::endl;

                } else if (event.key.code == sf::Keyboard::R) {
                    isRunning = true;
                    std::cout << "Simulation Resumed." << std::endl;
                }
            }
        }


        if(!isRunning)
            continue;

        get_all_velocity_magnitudes(particles, vels_curr);


        // Measure time elapsed since last frame
        sf::Time elapsed = clock.restart();
        float fps = 1.f / elapsed.asSeconds();
        // Update FPS text
        std::stringstream ss;
        ss << "Iter: "<< iter<<", FPS: " << fps;
        fpsText.setString(ss.str());

        if(iter % 1 == 0){ //hoombd-blue does it every 9th step
            rebuild_cell_list(particles, cell_list_idx, nblocks, nthreads);
        }

        // Compute forces
        vels_prev = vels_curr;
        collision<<<nblocks, nthreads>>>(particles, cell_list_idx);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Change after Collision:\n");
        get_all_velocity_magnitudes(particles, vels_curr);
        for(int i=0; i<N; i++){
            double force = (vels_curr[i] - vels_prev[i])  / dt;
            std::cout<<force<<", ";
        }
        std::cout<<std::endl;

        apply_forces<<<nblocks, nthreads>>>(particles, d_rand_states, sqrt_dt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        vels_prev = vels_curr;
        get_all_velocity_magnitudes(particles, vels_curr);
        printf("Change after Forces:\n");
        for(int i=0; i<N; i++){
            double force = (vels_curr[i] - vels_prev[i])  / dt;
            std::cout<<force<<", ";
        }
        std::cout<<std::endl<<std::endl;
        
        update_positions<<<nblocks, nthreads>>>(particles);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


        // Draw particles
        window.clear();
        for (int i=0; i<N; i++) {
            Particle particle = particles[i];
            shapes[i].setPosition(particle.x, particle.y);
            shapes[i].setFillColor(colors[particle.col]);
            window.draw(shapes[i]);
        }
        window.draw(fpsText);
        window.display();
    }
    
}




#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <curand_kernel.h>


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

const double dt = 0.05; // Time step
const double T = 9.0; // Temperature
const double GAMMA = 1.0; // Drag coefficient
const double mass = 1.0; // Mass of particles
const int steps = 10000; // Number of simulation steps

//Sim Box parameters
const int windowWidth = 800;
const int windowHeight = 600;

const sf::Color colors[4] = {sf::Color::Green, sf::Color::Red, sf::Color::Blue, sf::Color::Yellow};

struct Particle {
    double x = 0;
    double y = 0;
    double vx = 0;
    double vy = 0;

    sf::CircleShape shape;


    DEVICE Particle(float x, float y, const sf::Color col) : x(x), y(y) {
        shape.setRadius(RADIUS);
        shape.setPosition(x, y);
        shape.setFillColor(col);

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
        shape.setPosition(x, y);
    }

};

__global__ void render_init(RNG *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) 
        return;

    // TODO: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984, i, 0, &rand_state[i]);
}


// Compute collision forces based on Lennard-Jones potential
DEVICE void compute_collision_force(Particle& p1, Particle& p2, double& fx, double& fy) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx * dx + dy * dy;
    if(r2 < .1) { 
        fx = 0.0;
        fy = 0.0;
        return;
    }

    double r = std::sqrt(r2);
    
    // Lennard-Jones force magnitude
    double r_inv = sigma / r;
    double r_inv6 = r_inv * r_inv * r_inv * r_inv * r_inv * r_inv;
    double r_inv12 = r_inv6 * r_inv6;
    double f_magnitude = 4.0 * epsilon * ( r_inv12 - r_inv6) / r;

    fx = f_magnitude * dx / r;
    fy = f_magnitude * dy / r;
}

__global__ void collision(Particle *particles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    // FIXME: This is a O(N^2) algorithm
    for(int j = 0; j < N; j++){
        if(i == j)
            continue;
        double fx, fy;
        compute_collision_force(particles[i], particles[j], fx, fy);
        particles[i].vx += fx * dt / mass;
        particles[i].vy += fy * dt / mass;
    }

}

template <typename RNG>
__global__ void apply_forces(Particle *particles, RNG* rand_state){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;
    particles[i] = p;

    // Apply random force
    RNG local_rand_state = rand_state[i];
    p.vx += curand_uniform(&local_rand_state);
    p.vy += curand_uniform(&local_rand_state);
    rand_state[i] = local_rand_state;
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


    // Random number generator setup
    RNG *d_rand_states;
    checkCudaErrors(cudaMalloc((void **)&d_rand_states, N*sizeof(RNG)));


    // allocate FB
    Particle *particles;
    checkCudaErrors(cudaMallocManaged((void **)&particles, N * sizeof(Particle)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    render_init<<<nblocks, nthreads>>>(d_rand_states);

    // Set up SFML window
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Brownian Dynamics Simulation");
    window.setFramerateLimit(60);

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

    // Simulation loop
    int iter = 0;
    while (window.isOpen()) {
        iter++;
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        // Measure time elapsed since last frame
        sf::Time elapsed = clock.restart();
        float fps = 1.f / elapsed.asSeconds();
        // Update FPS text
        std::stringstream ss;
        ss << "Iter: "<< iter<<", FPS: " << fps;
        fpsText.setString(ss.str());

        // Compute forces

        collision<<<nblocks, nthreads>>>(particles);

        apply_forces<<<nblocks, nthreads>>>(particles, d_rand_states);
        update_positions<<<nblocks, nthreads>>>(particles);
        
        // Draw particles
        window.clear();
        for (int i=0; i<N; i++) {
            Particle particle = particles[i];
            window.draw(particle.shape);
        }
        window.draw(fpsText);
        window.display();
    }
    
}
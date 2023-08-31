#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include <sstream>


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


    Particle(float x, float y, int col_id) : x(x), y(y) {
        shape.setRadius(RADIUS);
        shape.setPosition(x, y);
        shape.setFillColor(colors[col_id]);

    }

    void update(float dx, float dy) {
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



// Compute collision forces based on Lennard-Jones potential
void compute_collision_force(Particle& p1, Particle& p2, double& fx, double& fy) {
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

void collision(int num_cells_x, int num_cells_y, 
    std::vector<Particle>& particles, std::vector<std::vector<std::vector<int>>>& cells){
    for (int i = 0; i < num_cells_x; ++i) {
        for (int j = 0; j < num_cells_y; ++j) {
            // Check interactions within this cell and neighboring cells
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    // Skip cells outside the simulation area
                    if (ni < 0 || ni >= num_cells_x || nj < 0 || nj >= num_cells_y) {
                        continue;
                    }

                    for (int p1 : cells[i][j]) {
                        for (int p2 : cells[ni][nj]) {
                            if (p1 != p2) {
                                double fx, fy;
                                compute_collision_force(particles[p1], particles[p2], fx, fy);
                                particles[p1].vx += fx;
                                particles[p1].vy += fy;
                                particles[p2].vx -= fx;
                                particles[p2].vy -= fy;
                            }
                        }
                    }
                }
            }
        }
    }
}

void assign_cells(std::vector<Particle>& particles, std::vector<std::vector<std::vector<int>>>& cells){
    const double cell_size = cutoff_distance;
    int max_cells_x = cells.size();
    int max_cells_y = cells[0].size();

    for(auto& cell : cells)
        for(auto& row : cell)
            row.clear();

    for (int i = 0; i < N; ++i) {        
        int cell_x = static_cast<int>((particles[i].x - 0) / cell_size);
        int cell_y = static_cast<int>((particles[i].y - 0) / cell_size);
        
        if(cell_x >= 0 && cell_x < max_cells_x && cell_y >= 0 && cell_y < max_cells_y) {
            cells[cell_x][cell_y].push_back(i);
        } else {
            // Handle the out-of-bounds condition.
            std::cout << "Particle out of simulation box:"<<cell_x<<","<<cell_y << std::endl;
        }
    }
    
}

//
template <typename RNG>
void apply_forces(std::vector<Particle>& particles, RNG& gen, std::normal_distribution<>& d){
    for (auto& p : particles) {
        // Apply drag force
        p.vx -= GAMMA / mass * p.vx * dt;
        p.vy -= GAMMA / mass * p.vy * dt;

        // Apply random force
        p.vx += d(gen);
        p.vy += d(gen);
    }
}

void update_positions(std::vector<Particle>& particles){
    for (auto& p : particles) {
        // Check for collisions with box boundaries
        if (p.x - RADIUS < 0 || p.x + RADIUS > windowWidth) {
            p.vx *= -1;
        }
        if (p.y - RADIUS < 0 || p.y + RADIUS > windowHeight) {
            p.vy *= -1;
        }
        // Update positions
        p.update(p.vx * dt, p.vy * dt);

    }
}

int main() {


    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force

    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, sqrt_dt);
    std::uniform_real_distribution<> init_x(0, windowWidth);
    std::uniform_real_distribution<> init_y(0, windowHeight);


    // Parameters for simulation
    const double cell_size = cutoff_distance;
    const int num_cells_x = windowWidth / cell_size + 1;
    const int num_cells_y = windowHeight / cell_size + 1;
    std::cout<<num_cells_x<<","<<num_cells_y<< std::endl;

    std::vector<std::vector<std::vector<int>>> cells(num_cells_x, std::vector<std::vector<int>>(num_cells_y));

    // Initialize particles
    std::vector<Particle> particles;
    for (int i = 0; i < N; ++i) 
        particles.emplace_back(init_x(gen), init_y(gen), i%4);
            

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

        assign_cells(particles, cells); // TODO: optimize this call
        collision(num_cells_x, num_cells_y, particles, cells);
        apply_forces(particles, gen, d);
        update_positions(particles);
        
        // Draw particles
        window.clear();
        for (const Particle& particle : particles) {
            window.draw(particle.shape);
        }
        window.draw(fpsText);
        window.display();
    }

    return 0;
}

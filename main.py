import numpy as np
from numba import cuda, njit
import math
import time

# Rastrigin Fitness Function (CPU-based, for sequential execution)
def rastrigin(individual):
    n = len(individual)
    return 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in individual)

# Rastrigin Fitness Function (GPU Kernel)
@cuda.jit
def rastrigin_gpu(population, fitness_values, dimensions):
    i = cuda.grid(1)  # Get thread index
    if i < population.shape[0]:  # Ensure thread is within bounds
        total = 10 * dimensions
        for j in range(dimensions):
            x = population[i, j]
            total += x**2 - 10 * math.cos(2 * math.pi * x)  # Use math.cos for GPU
        fitness_values[i] = total

# Initialize Population
def initialize_population(pop_size, dimensions, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))

# Selection: Roulette Wheel
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]

# Crossover: Single-point crossover
def crossover(parents, crossover_rate=0.8):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parents[i]))
            child1 = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
            child2 = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i+1]])
    return np.array(offspring)

# Mutation: Random mutation
def mutation(population, mutation_rate=0.1, lower_bound=-5.12, upper_bound=5.12):
    for individual in population:
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(individual))
            individual[idx] = np.random.uniform(lower_bound, upper_bound)
    return population

# Fitness Evaluation on GPU
def parallel_fitness_gpu(population, dimensions):
    pop_size = population.shape[0]

    # Allocate memory on the GPU
    d_population = cuda.to_device(population)
    d_fitness_values = cuda.device_array(pop_size)

    # Configure grid/block size
    threads_per_block = 256
    blocks_per_grid = (pop_size + threads_per_block - 1) // threads_per_block

    # Launch kernel
    rastrigin_gpu[blocks_per_grid, threads_per_block](d_population, d_fitness_values, dimensions)

    # Copy results back to host
    return d_fitness_values.copy_to_host()

# Genetic Algorithm
def genetic_algorithm(pop_size, dimensions, generations, use_gpu=False):
    lower_bound, upper_bound = -5.12, 5.12
    population = initialize_population(pop_size, dimensions, lower_bound, upper_bound)
    
    for generation in range(generations):
        # Fitness evaluation (sequential or GPU)
        if use_gpu:
            fitness_values = parallel_fitness_gpu(population, dimensions)
        else:
            fitness_values = [rastrigin(ind) for ind in population]
        
        # GA operations
        selected = selection(population, fitness_values)
        offspring = crossover(selected)
        population = mutation(offspring, lower_bound=lower_bound, upper_bound=upper_bound)
    
    # Final evaluation
    fitness_values = parallel_fitness_gpu(population, dimensions) if use_gpu else [rastrigin(ind) for ind in population]
    best_individual = population[np.argmin(fitness_values)]
    best_fitness = min(fitness_values)
    return best_individual, best_fitness

# Main Execution
if __name__ == "__main__":
    pop_size = 1000  # Population size
    dimensions = 50  # Dimensions of the problem
    generations = 100  # Number of generations

    # Sequential Execution
    print("Running Sequential Genetic Algorithm...")
    start_time = time.time()
    best_seq, fitness_seq = genetic_algorithm(pop_size, dimensions, generations, use_gpu=False)
    seq_time = time.time() - start_time
    print(f"Best Solution (Sequential): {best_seq}, Fitness: {fitness_seq}, Time: {seq_time:.2f} seconds")

    # GPU Execution
    print("\nRunning GPU-Accelerated Genetic Algorithm...")
    start_time = time.time()
    best_gpu, fitness_gpu = genetic_algorithm(pop_size, dimensions, generations, use_gpu=True)
    gpu_time = time.time() - start_time
    print(f"Best Solution (GPU): {best_gpu}, Fitness: {fitness_gpu}, Time: {gpu_time:.2f} seconds")

    # Efficiency Comparison
    print(f"\nEfficiency Comparison:\nSequential Time: {seq_time:.2f} seconds\nGPU Time: {gpu_time:.2f} seconds")

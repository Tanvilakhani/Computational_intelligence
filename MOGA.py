import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap.tools import sortNondominated

# ==== Step 1: Parse scheduling data ====
def parse_scheduling_data(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    multi_skilling_level = 66  # fixed
    problem_type = int(lines[0].split('=')[1])
    num_jobs = int(lines[1].split('=')[1])

    job_lines = lines[2:2 + num_jobs]
    job_times = [tuple(map(int, line.split())) for line in job_lines]

    qualifications_header_index = 2 + num_jobs
    num_shifts = int(lines[qualifications_header_index].split('=')[1])

    qualification_lines = lines[qualifications_header_index + 1:]
    shift_qualifications = []
    for line in qualification_lines:
        parts = line.split(':')[1].strip()
        qualifications = [int(x) for x in parts.split()]
        shift_qualifications.append(qualifications)

    return {
        "multi_skilling_level": multi_skilling_level,
        "problem_type": problem_type,
        "num_jobs": num_jobs,
        "job_times": job_times,
        "num_shifts": num_shifts,
        "shift_qualifications": shift_qualifications
    }

# ==== Step 2: Main algorithm ====
def main(filename):
    print(f"Running on file: {filename}")

    random.seed(42)

    # Load data
    data = parse_scheduling_data(filename)
    NUM_JOBS = data["num_jobs"]
    NUM_SHIFTS = data["num_shifts"]
    SHIFT_QUALIFICATIONS = data["shift_qualifications"]
    JOB_TIMES = data["job_times"]

    SHIFT_IDS = list(range(NUM_SHIFTS))

    # Create Fitness and Individual
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize overlap, maximize coverage
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Create Toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_shift", random.choice, SHIFT_IDS)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_shift, n=NUM_JOBS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Objective Function
    def evaluate(individual):
        overlap_penalty = 0
        for i in range(NUM_JOBS):
            for j in range(i + 1, NUM_JOBS):
                if individual[i] == individual[j]:  # Same shift
                    start_i, end_i = JOB_TIMES[i]
                    start_j, end_j = JOB_TIMES[j]
                    if not (end_i <= start_j or end_j <= start_i):  # Overlap
                        overlap_penalty += 1

        coverage_score = 0
        for shift_id in range(NUM_SHIFTS):
            required_skills = SHIFT_QUALIFICATIONS[shift_id]
            assigned_jobs = [i for i, shift in enumerate(individual) if shift == shift_id]

            for job in assigned_jobs:
                job_skill = job % 5  # Simplified skill assignment
                if job_skill in required_skills:
                    coverage_score += 1

        return overlap_penalty, coverage_score

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_SHIFTS-1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    # Parameters
    population_size = 100
    generations = 50
    crossover_probability = 0.8
    mutation_probability = 0.2

    # Initial population
    population = toolbox.population(n=population_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolution
    pareto_front = []

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        population = toolbox.select(offspring + population, k=len(population))
        pareto_front += population

    # Final non-dominated solutions
    non_dominated_solutions = sortNondominated(pareto_front, len(pareto_front), first_front_only=True)[0]
    non_dominated_fitness = np.array([ind.fitness.values for ind in non_dominated_solutions])

    # ==== Step 3: Plot Pareto Front ====

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(non_dominated_fitness[:, 0], non_dominated_fitness[:, 1], c=np.arange(len(non_dominated_fitness)), cmap='viridis')
    ax.set_xlabel('Objective 1: Minimize Overlap')
    ax.set_ylabel('Objective 2: Maximize Coverage')
    ax.set_zlabel('Solution Index')
    ax.set_title(f'3D Pareto Front\n{filename}')
    plt.show()    # <-- This will block until you close the plot

    # ==== Step 4: Plot Heatmap ====
    job_shift_matrix = np.zeros((NUM_JOBS, NUM_SHIFTS))
    best_solution = non_dominated_solutions[0]

    for job_idx, shift in enumerate(best_solution):
        job_shift_matrix[job_idx, shift] = 1

    plt.figure(figsize=(10, 6))
    plt.imshow(job_shift_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Assignment (1 = Assigned)')
    plt.title(f'Job Assignments to Shifts (Heatmap)\n{filename}')
    plt.xlabel('Shifts')
    plt.ylabel('Jobs')
    plt.xticks(ticks=np.arange(NUM_SHIFTS), labels=[f"Shift {i}" for i in range(NUM_SHIFTS)])
    plt.yticks(ticks=np.arange(NUM_JOBS), labels=[f"Job {i}" for i in range(NUM_JOBS)])
    plt.show()    # <-- Again, will wait until you close the heatmap window



# ==== Step 5: Entry Point ====
if __name__ == "__main__":

    files = [
        "ptask_dataset/data_1_23_40_66.dat",
        "ptask_dataset/data_2_24_40_33.dat",
        "ptask_dataset/data_3_25_40_66.dat"
    ]

    for file in files:
        main(file)

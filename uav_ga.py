"""
UAV Disaster-Surveillance Strategy Engine
Run: python uav_ga.py
Creates: fitness_convergence.png, uav_flight_path.png, coverage_heatmap.png
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = 20
START = (1, 1)
GOAL = (18, 18)
N_WAYPOINTS = 6
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.18
ELITE_COUNT = 4
TOURNAMENT_SIZE = 4
MAX_FLIGHT_RANGE = 58.0

# Map values: 0=open, 1=no-fly zone, 2=hazard zone.
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
grid[4:8, 5:8] = 1
grid[11:15, 11:14] = 1
grid[2:5, 13:17] = 2
grid[13:17, 3:7] = 2
grid[8:11, 15:18] = 2

# High-priority disaster targets (x, y); these are simulated, not real locations.
TARGETS = [(5, 12), (9, 5), (15, 9), (16, 16), (7, 17), (12, 3)]
SENSOR_LEVELS = [0, 1, 2]       # off, medium, high
SPEED_LEVELS = [1.0, 1.5, 2.0]  # relative speed settings


class Chromosome:
    """A candidate route plus sensor and speed settings for each route segment."""
    def __init__(self, points=None, sensors=None, speeds=None):
        self.points = points if points is not None else [
            (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
            for _ in range(N_WAYPOINTS)
        ]
        self.sensors = sensors if sensors is not None else [
            random.choice(SENSOR_LEVELS) for _ in range(N_WAYPOINTS + 1)
        ]
        self.speeds = speeds if speeds is not None else [
            random.choice(SPEED_LEVELS) for _ in range(N_WAYPOINTS + 1)
        ]
        self.fitness = None

    def copy(self):
        return Chromosome(self.points[:], self.sensors[:], self.speeds[:])


def clamp_point(point):
    return (max(0, min(GRID_SIZE - 1, int(point[0]))),
            max(0, min(GRID_SIZE - 1, int(point[1]))))


def full_route(chromosome):
    return [START] + [clamp_point(p) for p in chromosome.points] + [GOAL]


def sample_segment(a, b):
    """Return grid samples along a straight segment between two (x, y) points."""
    distance = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    if distance == 0:
        return [a]
    samples = []
    for i in range(distance + 1):
        t = i / distance
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        samples.append((x, y))
    return samples


def evaluate(chromosome):
    route = full_route(chromosome)
    total_distance = 0.0
    hazard_cost = 0.0
    no_fly_hits = 0
    visited = []
    energy = 0.0
    flight_time = 0.0
    coverage = np.zeros_like(grid, dtype=float)

    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        distance = math.dist(a, b)
        total_distance += distance
        sensor = chromosome.sensors[i]
        speed = chromosome.speeds[i]
        energy += distance * (0.8 + 0.55 * sensor + 0.15 * speed * speed)
        flight_time += distance / speed
        segment = sample_segment(a, b)
        visited.extend(segment)
        for x, y in segment:
            if grid[y, x] == 1:
                no_fly_hits += 1
            elif grid[y, x] == 2:
                hazard_cost += 1.0
            # Sensor power expands the effective coverage radius.
            radius = 0.7 + 0.65 * sensor
            for tx, ty in TARGETS:
                if math.dist((x, y), (tx, ty)) <= radius:
                    coverage[ty, tx] = max(coverage[ty, tx], (sensor + 1) / 3)

    target_coverage = sum(coverage[ty, tx] for tx, ty in TARGETS) / len(TARGETS)
    # Reward surveying more distinct cells, but prioritize high-value targets.
    unique_cells = len(set(visited))
    area_coverage = min(1.0, unique_cells / 45.0)
    detection = target_coverage
    range_penalty = max(0.0, total_distance - MAX_FLIGHT_RANGE)

    # Weighted multi-objective score: maximize coverage/detection; minimize costs.
    score = (
        45.0 * target_coverage
        + 18.0 * area_coverage
        - 0.35 * energy
        - 0.45 * flight_time
        - 3.5 * hazard_cost
        - 12.0 * no_fly_hits
        - 2.0 * range_penalty
    )
    chromosome.fitness = score
    return score, route, coverage, {
        "distance": total_distance, "energy": energy, "time": flight_time,
        "hazards": hazard_cost, "no_fly_hits": no_fly_hits,
        "target_coverage": target_coverage, "area_coverage": area_coverage
    }


def initialize_population():
    return [Chromosome() for _ in range(POPULATION_SIZE)]


def tournament_selection(population):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return max(contestants, key=lambda item: item.fitness)


def crossover(parent1, parent2):
    """Custom segment crossover: exchange route, sensor and speed genes together."""
    cut1 = random.randint(1, N_WAYPOINTS - 1)
    points = parent1.points[:cut1] + parent2.points[cut1:]
    cut2 = random.randint(1, N_WAYPOINTS)
    sensors = parent1.sensors[:cut2] + parent2.sensors[cut2:]
    speeds = parent1.speeds[:cut2] + parent2.speeds[cut2:]
    child = Chromosome(points, sensors, speeds)
    child.points = [clamp_point(p) for p in child.points]
    return child


def mutate(chromosome):
    """Coordinate mutation plus sensor/speed changes to preserve diversity."""
    for i, (x, y) in enumerate(chromosome.points):
        if random.random() < MUTATION_RATE:
            chromosome.points[i] = clamp_point((
                x + random.randint(-4, 4), y + random.randint(-4, 4)
            ))
    for i in range(len(chromosome.sensors)):
        if random.random() < MUTATION_RATE:
            chromosome.sensors[i] = random.choice(SENSOR_LEVELS)
        if random.random() < MUTATION_RATE:
            chromosome.speeds[i] = random.choice(SPEED_LEVELS)
    chromosome.fitness = None
    return chromosome


def run_ga():
    population = initialize_population()
    best_history, average_history = [], []
    best_ever = None

    for generation in range(GENERATIONS):
        for individual in population:
            if individual.fitness is None:
                evaluate(individual)
        population.sort(key=lambda item: item.fitness, reverse=True)
        if best_ever is None or population[0].fitness > best_ever.fitness:
            best_ever = population[0].copy()
            best_ever.fitness = population[0].fitness
        best_history.append(population[0].fitness)
        average_history.append(float(np.mean([p.fitness for p in population])))

        next_population = [p.copy() for p in population[:ELITE_COUNT]]
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = mutate(crossover(p1, p2))
            next_population.append(child)
        population = next_population

    evaluate(best_ever)
    return best_ever, best_history, average_history


def save_fitness_plot(best_history, average_history):
    plt.figure(figsize=(8, 5))
    plt.plot(best_history, label="Best fitness")
    plt.plot(average_history, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (higher is better)")
    plt.title("Genetic Algorithm Fitness Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitness_convergence.png", dpi=160)
    plt.close()


def save_path_plot(chromosome):
    _, route, _, stats = evaluate(chromosome)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Display no-fly zones and hazards as shaded map cells.
    ax.imshow(grid, origin="lower", cmap="YlOrRd", alpha=0.55,
              extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
              vmin=0, vmax=2)
    xs, ys = zip(*route)
    ax.plot(xs, ys, "-o", linewidth=2, markersize=4, label="Evolved UAV path")
    ax.scatter(*START, marker="s", s=100, label="Start")
    ax.scatter(*GOAL, marker="*", s=180, label="Goal")
    tx, ty = zip(*TARGETS)
    ax.scatter(tx, ty, marker="X", s=80, label="Priority targets")
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(0, GRID_SIZE, 2))
    ax.set_yticks(range(0, GRID_SIZE, 2))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Evolved UAV Route (red cells = restricted/hazard areas)")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("uav_flight_path.png", dpi=160)
    plt.close(fig)
    return stats


def save_coverage_plot(chromosome):
    _, route, coverage, _ = evaluate(chromosome)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(coverage, origin="lower", cmap="viridis", vmin=0, vmax=1)
    tx, ty = zip(*TARGETS)
    ax.scatter(tx, ty, marker="X", s=90, edgecolors="white", label="Priority targets")
    xs, ys = zip(*route)
    ax.plot(xs, ys, "w--", linewidth=1.5, label="Flight path")
    ax.set_title("Sensor Coverage Heatmap")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("coverage_heatmap.png", dpi=160)
    plt.close(fig)


def main():
    print("Running UAV Genetic Algorithm simulation...")
    best, best_history, average_history = run_ga()
    stats = save_path_plot(best)
    save_fitness_plot(best_history, average_history)
    save_coverage_plot(best)
    print("\nBest candidate summary")
    print("----------------------")
    print(f"Fitness: {best.fitness:.2f}")
    print(f"Route: {full_route(best)}")
    print(f"Distance: {stats['distance']:.2f} (limit {MAX_FLIGHT_RANGE:.1f})")
    print(f"Energy estimate: {stats['energy']:.2f}")
    print(f"Flight-time estimate: {stats['time']:.2f}")
    print(f"Target coverage score: {stats['target_coverage']:.2f}")
    print(f"Area coverage score: {stats['area_coverage']:.2f}")
    print(f"Hazard samples: {stats['hazards']:.0f}")
    print(f"No-fly-zone samples: {stats['no_fly_hits']}")
    print("\nSaved diagrams:")
    print("  fitness_convergence.png")
    print("  uav_flight_path.png")
    print("  coverage_heatmap.png")
    print("\nNote: This is a simplified educational simulation, not real flight-control software.")

def save_path_plot(chromosome):
    _, route, _, stats = evaluate(chromosome)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Display no-fly zones and hazards as shaded map cells.
    ax.imshow(grid, origin="lower", cmap="YlOrRd", alpha=0.55,
              extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
              vmin=0, vmax=2)
    xs, ys = zip(*route)
    ax.plot(xs, ys, "-o", linewidth=2, markersize=4, label="Evolved UAV path")
    ax.scatter(*START, color="green", marker="s", s=100, label="Start")
    ax.scatter(*GOAL, color="red", marker="*", s=180, label="Goal")
    tx, ty = zip(*TARGETS)
    ax.scatter(tx, ty, color="purple", marker="X", s=80, label="Priority targets")
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(0, GRID_SIZE, 2))
    ax.set_yticks(range(0, GRID_SIZE, 2))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Evolved UAV Flight Path")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("uav_flight_path.png", dpi=160)
    plt.close()

def save_heatmap_plot(chromosome):
    _, _, coverage, _ = evaluate(chromosome)
    plt.figure(figsize=(7, 7))
    plt.imshow(coverage, origin="lower", cmap="viridis", extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5))
    plt.colorbar(label="Sensor Coverage Intensity")
    tx, ty = zip(*TARGETS)
    plt.scatter(tx, ty, color="red", marker="X", s=80, label="Targets")
    plt.title("UAV Sensor Coverage Heatmap")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_heatmap.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    print("Running UAV Disaster-Surveillance GA Engine...")
    best_sol, b_hist, a_hist = run_ga()
    
    print("Saving plots to workspace...")
    save_fitness_plot(b_hist, a_hist)
    save_path_plot(best_sol)
    save_heatmap_plot(best_sol)
    print("Done! Check your folder or Port 5501 for the .png files.")
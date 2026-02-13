import numpy as np


class Particle:
    def __init__(self, bounds):
        self.pos = np.array([
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[0], bounds[1])
        ])
        self.velocity = np.array([0.0, 0.0])
        self.best_pos = self.pos.copy()
        self.best_value = float('inf')

    def enforce_bounds(self, bounds):
        self.pos[0] = np.clip(self.pos[0], bounds[0], bounds[1])
        self.pos[1] = np.clip(self.pos[1], bounds[0], bounds[1])


def run_pso(func, bounds, config):
    n_particles = config['n_particles']
    n_iterations = config['n_iterations']
    w = config['inertia']
    c1 = config['c_local']
    c2 = config['c_global']

    swarm = [Particle(bounds) for _ in range(n_particles)]
    global_best_pos = None
    global_best_value = float('inf')

    best_history = []

    for _ in range(n_iterations):
        for particle in swarm:
            value = func(particle.pos[0], particle.pos[1])

            if value < particle.best_value:
                particle.best_value = value
                particle.best_pos = particle.pos.copy()

            if value < global_best_value:
                global_best_value = value
                global_best_pos = particle.pos.copy()

        best_history.append(global_best_value)

        for particle in swarm:
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)

            inertia = w * particle.velocity
            cognitive = c1 * r1 * (particle.best_pos - particle.pos)
            social = c2 * r2 * (global_best_pos - particle.pos)

            particle.velocity = inertia + cognitive + social
            particle.pos = particle.pos + particle.velocity
            particle.enforce_bounds(bounds)

    final_positions = [p.pos.copy() for p in swarm]
    return global_best_pos, global_best_value, final_positions, best_history

import numpy as np
import random
import math


class VRPTW_ACO_Hybrid:
    def __init__(self, customers, capacity, m=40, T=100, alpha=1.0, beta=5.0, rho=0.5, p_random=0.1):
        self.customers = customers
        self.n_nodes = len(customers)
        self.capacity = capacity
        self.m = m
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.p_random = p_random

        self.distance_matrix = self.calculate_distance_matrix()
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes))

    def calculate_distance_matrix(self):
        dist_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    c1, c2 = self.customers[i], self.customers[j]
                    dist = math.sqrt((c1['x'] - c2['x']) ** 2 + (c1['y'] - c2['y']) ** 2)
                    dist_matrix[i][j] = dist
        return dist_matrix

    def select_next_node(self, current_node, unvisited, current_time, current_load):
        candidates = []
        for node in unvisited:
            dist = self.distance_matrix[current_node][node]
            arrival_time = max(self.customers[node]['ready'], current_time + dist)
            if (arrival_time <= self.customers[node]['due'] and
                    current_load + self.customers[node]['demand'] <= self.capacity):
                candidates.append(node)

        if not candidates:
            return None

        if random.random() < self.p_random:
            return random.choice(candidates)

        probabilities = []
        denominator = 0.0
        for node in candidates:
            dist = self.distance_matrix[current_node][node]
            heuristic = 1.0 / (dist + 0.1 * (self.customers[node]['due'] - current_time))
            pheromone = self.pheromone_matrix[current_node][node]
            prob_value = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob_value)
            denominator += prob_value

        if denominator == 0:
            return random.choice(candidates)

        probabilities = [p / denominator for p in probabilities]
        return np.random.choice(candidates, p=probabilities)

    def two_opt(self, route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if self.is_route_feasible(new_route):
                        if self.get_route_dist(new_route) < self.get_route_dist(best):
                            best = new_route
                            improved = True
            if not improved: break
        return best

    def is_route_feasible(self, r):
        t, l = 0, 0
        for i in range(len(r) - 1):
            dist = self.distance_matrix[r[i]][r[i + 1]]
            t = max(self.customers[r[i + 1]]['ready'], t + dist)
            if t > self.customers[r[i + 1]]['due']: return False
            l += self.customers[r[i + 1]]['demand']
            if l > self.capacity: return False
            t += self.customers[r[i + 1]]['service']
        return True

    def get_route_dist(self, r):
        return sum(self.distance_matrix[r[i]][r[i + 1]] for i in range(len(r) - 1))

    def update_pheromones(self, all_solutions_data):
        self.pheromone_matrix *= (1 - self.rho)

        for routes, total_dist, num_vehicles in all_solutions_data:
            deposit = 1.0 / (total_dist * num_vehicles)

            for route in routes:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    self.pheromone_matrix[u][v] += deposit
                    self.pheromone_matrix[v][u] += deposit

    def run(self):
        best_global_solution = None
        best_global_score = (float('inf'), float('inf'))
        history = []

        for iteration in range(self.T):
            iteration_solutions = []
            for _ in range(self.m):
                unvisited = set(range(1, self.n_nodes))
                ant_routes = []
                exceeded_limit = False  # Flaga limitu

                while unvisited:
                    if len(ant_routes) >= 25:
                        print(f"BŁĄD: Przekroczono limit 25 pojazdów w iteracji {iteration}. Przerywam tę mrówkę.")
                        exceeded_limit = True
                        break

                    route = [0]
                    curr_node, time, load = 0, 0, 0
                    while True:
                        next_node = self.select_next_node(curr_node, unvisited, time, load)
                        if next_node is None: break

                        dist = self.distance_matrix[curr_node][next_node]
                        time = max(self.customers[next_node]['ready'], time + dist)
                        time += self.customers[next_node]['service']
                        load += self.customers[next_node]['demand']

                        route.append(int(next_node))
                        unvisited.remove(next_node)
                        curr_node = next_node

                    route.append(0)
                    route = self.two_opt(route)
                    ant_routes.append(route)

                if exceeded_limit:
                    total_dist = float('inf')
                    num_vehicles = float('inf')
                else:
                    total_dist = sum(self.get_route_dist(r) for r in ant_routes)
                    num_vehicles = len(ant_routes)

                current_score = (num_vehicles, total_dist)
                iteration_solutions.append((ant_routes, total_dist, num_vehicles))

                if current_score < best_global_score:
                    best_global_score = current_score
                    best_global_solution = ant_routes

            self.update_pheromones(iteration_solutions)
            history.append(best_global_score)

            print(f"Iteracja {iteration}: Pojazdy = {best_global_score[0]}, Dystans = {best_global_score[1]:.2f}")

        return best_global_solution, best_global_score, history


def load_solomon(filename):
    customers = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        capacity = int(lines[4].split()[1])
        for line in lines[9:]:
            parts = list(map(int, line.split()))
            if not parts: continue
            customers.append({
                'id': parts[0], 'x': parts[1], 'y': parts[2],
                'demand': parts[3], 'ready': parts[4], 'due': parts[5], 'service': parts[6]
            })
    return customers, capacity
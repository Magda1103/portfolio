import numpy as np
import random


class AntColonyOptimizer:
    def __init__(self, locations, m=20, T=100, alpha=1.0, beta=2.0, rho=0.5, p_random=0.0):
        """
        Inicjalizacja algorytmu z parametrami zgodnymi z treścią zadania.
        :param m: Liczebność mrówek (dawniej n_ants)
        :param T: Liczba iteracji (dawniej n_iterations)
        :param alpha: Współczynnik wpływu feromonów
        :param beta: Współczynnik wpływu heurystyki
        :param rho: Współczynnik wyparowywania feromonów
        :param p_random: Prawdopodobieństwo wyboru losowej atrakcji
        """
        self.locations = locations
        self.n_nodes = len(locations)
        self.m = m  # Liczba mrówek
        self.T = T  # Liczba iteracji
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.p_random = p_random

        self.distance_matrix = self.calculate_distance_matrix()
        # Inicjalizacja feromonów wartością 1.0
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes))

    def calculate_distance_matrix(self):
        dist_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    dist = np.sqrt((self.locations[i][1] - self.locations[j][1]) ** 2 +
                                   (self.locations[i][2] - self.locations[j][2]) ** 2)
                    dist_matrix[i][j] = dist
        return dist_matrix

    def select_next_node(self, current_node, visited):
        unvisited = [i for i in range(self.n_nodes) if i not in visited]

        # 1. Losowość (p_random)
        if random.random() < self.p_random:
            return random.choice(unvisited)

        probabilities = []
        denominator = 0.0

        for node in unvisited:
            dist = self.distance_matrix[current_node][node]

            # Zabezpieczenie dla miast o tych samych współrzędnych (d=0)
            if dist == 0:
                heuristic = 1e10
            else:
                heuristic = 1.0 / dist

            pheromone = self.pheromone_matrix[current_node][node]

            prob_value = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob_value)
            denominator += prob_value

        if denominator == 0:
            return random.choice(unvisited)

        probabilities = [p / denominator for p in probabilities]

        return np.random.choice(unvisited, p=probabilities)

    def run(self):
        best_global_route = None
        best_global_distance = float('inf')
        history_best = []

        # Pętla po iteracjach (T)
        for iteration in range(self.T):
            all_routes = []
            all_distances = []

            # Konstrukcja tras przez mrówki (m)
            for _ in range(self.m):
                start_node = random.randint(0, self.n_nodes - 1)
                route = [start_node]
                visited = {start_node}

                current_node = start_node
                while len(route) < self.n_nodes:
                    next_node = self.select_next_node(current_node, visited)
                    route.append(next_node)
                    visited.add(next_node)
                    current_node = next_node

                # Obliczenie długości trasy
                total_dist = 0
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i]][route[i + 1]]
                total_dist += self.distance_matrix[route[-1]][route[0]]

                all_routes.append(route)
                all_distances.append(total_dist)

                if total_dist < best_global_distance:
                    best_global_distance = total_dist
                    best_global_route = route

            # Aktualizacja feromonów
            self.update_pheromones(all_routes, all_distances)
            history_best.append(best_global_distance)

        return best_global_route, best_global_distance, history_best

    def update_pheromones(self, routes, distances):
        # Wyparowywanie (rho)
        self.pheromone_matrix *= (1 - self.rho)

        # Naparowywanie
        for route, dist in zip(routes, distances):
            deposit = 1.0 / dist
            for i in range(self.n_nodes - 1):
                u, v = route[i], route[i + 1]
                self.pheromone_matrix[u][v] += deposit
                self.pheromone_matrix[v][u] += deposit

            # Powrót do startu
            u, v = route[-1], route[0]
            self.pheromone_matrix[u][v] += deposit
            self.pheromone_matrix[v][u] += deposit
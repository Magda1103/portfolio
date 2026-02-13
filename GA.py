import random
import pandas as pd
import time


class GA:
    def __init__(self, pc, pm, n, t, load_capacity):
        self.pc = pc
        self.pm = pm
        self.n = n
        self.t = t
        self.load_capacity = load_capacity

        # Wczytywanie danych
        try:
            self.dataset = pd.read_excel('dane.xlsx')
        except FileNotFoundError:
            raise FileNotFoundError("Nie znaleziono pliku 'dane.xlsx'!")

        self.count_items = len(self.dataset)
        self.population = []

        # Optymalizacja (listy zamiast DataFrame w pętli dla szybkości)
        self.weights = self.dataset['Waga (kg)'].tolist()
        self.values = self.dataset['Wartość (zł)'].tolist()

    def generate_initial_population(self):
        self.population = []
        for item in range(self.n):
            actual_individual = []
            for gen in range(self.count_items):
                random_gen = random.choice([0, 1])
                actual_individual.append(random_gen)
            self.population.append(actual_individual)
        return self.population

    def get_population(self):
        return self.population

    def calculate_fitness(self, individual):
        sum_weight = 0
        sum_value = 0

        # Obliczamy wagę ORAZ wartość
        for gen_index in range(self.count_items):
            if individual[gen_index] == 1:
                sum_weight += self.weights[gen_index]
                sum_value += self.values[gen_index]  # Dodano sumowanie wartości

        # Jeśli przekroczono udźwig, fitness = 0 (kara)
        if sum_weight > self.load_capacity:
            return 0

        # Celem jest maksymalizacja WARTOŚCI, nie wagi
        return sum_value

    # --- METODY SELEKCJI ---

    def set_probability(self):
        p = self.get_population()
        fitness_values = [self.calculate_fitness(ind) for ind in p]
        sum_adaptation = sum(fitness_values)

        return [fit / sum_adaptation for fit in fitness_values]

    def roullete_selection(self, number_to_select):
        probability = self.set_probability()

        selected_parents = random.choices(self.population, weights=probability, k=number_to_select)
        return selected_parents

    def tournament_selection(self, number_to_select, tournament_size=3):
        selected_parents = []
        for _ in range(number_to_select):
            # 1. Losujemy k kandydatów z populacji
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            # 2. Wybieramy tego z najlepszym fitness
            winner = max(candidates, key=self.calculate_fitness)
            selected_parents.append(winner)
        return selected_parents

    # --- METODY KRZYŻOWANIA ---

    def one_point_crossover(self, parent_a, parent_b, crossover_point):
        children = []
        child_1 = parent_a[:crossover_point] + parent_b[crossover_point:]
        children.append(child_1)
        child_2 = parent_b[:crossover_point] + parent_a[crossover_point:]
        children.append(child_2)
        return children

    def two_point_crossover(self, parent_a, parent_b):
        children = []
        # Losujemy dwa różne punkty cięcia
        # range(1, len-1) zapewnia, że punkty nie są na skrajach
        points = random.sample(range(1, len(parent_a)), 2)
        p1, p2 = sorted(points)

            # Dziecko 1: A-B-A
        c1 = parent_a[:p1] + parent_b[p1:p2] + parent_a[p2:]
        children.append(c1)

            # Dziecko 2: B-A-B
        c2 = parent_b[:p1] + parent_a[p1:p2] + parent_b[p2:]
        children.append(c2)

        return children

    # --- MUTACJA ---

    def mutate(self, individual):
        random_index = random.randint(0, len(individual) - 1)
        if individual[random_index] == 1:
            individual[random_index] = 0
        else:
            individual[random_index] = 1
        return individual

    # --- WYKONANIE ALGORYTMU ---

    def execute_ga_with_stats(self, selection_method, crossover_method):
        start_time = time.time()
        best_global_fitness = 0
        best_global_individual = []
        history = []

        self.generate_initial_population()

        for generation in range(self.t):
            # Statystyki obecnego pokolenia
            current_fitnesses = [self.calculate_fitness(ind) for ind in self.population]
            best_current = max(current_fitnesses)
            worst_current = min(current_fitnesses)

            # Zabezpieczenie przed dzieleniem przez zero przy pustych listach (teoretycznie niemożliwe tu, ale bezpieczniej)
            if len(current_fitnesses) > 0:
                avg_current = sum(current_fitnesses) / len(current_fitnesses)
            else:
                avg_current = 0

            history.append({
                'generation': generation,
                'best': best_current,
                'worst': worst_current,
                'avg': avg_current
            })

            # Aktualizacja najlepszego globalnie
            if best_current > best_global_fitness:
                best_global_fitness = best_current
                best_ind_index = current_fitnesses.index(best_current)
                best_global_individual = self.population[best_ind_index][:]

            # 1. SELEKCJA (WYBÓR METODY)
            if selection_method == 'tournament':
                selected_parents = self.tournament_selection(self.n, tournament_size=3)
            else:
                selected_parents = self.roullete_selection(self.n)

            children = []

            # 2. KRZYŻOWANIE (Pętla po parach)
            for i in range(0, len(selected_parents), 2):
                parent_a = selected_parents[i]
                parent_b = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]

                if random.random() < self.pc:
                    # WYBÓR METODY KRZYŻOWANIA
                    if crossover_method == 'two_point':
                        new_kids = self.two_point_crossover(parent_a, parent_b)
                    else:
                        point = random.randint(1, self.count_items - 1)
                        new_kids = self.one_point_crossover(parent_a, parent_b, point)

                    children.extend(new_kids)
                else:
                    children.extend([parent_a, parent_b])

            # 3. MUTACJA
            for child in children:
                if random.random() < self.pm:
                    self.mutate(child)

            self.population = children[:self.n]

        end_time = time.time()
        execution_time = end_time - start_time

        return best_global_fitness, best_global_individual, execution_time, history
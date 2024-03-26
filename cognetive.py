import random
import numpy as np
from sklearn.metrics import accuracy_score

# Функция для оценки когнитрона
def evaluate_cognitron(cognitron, X_train, y_train):
    # Обучение когнитрона
    cognitron.fit(X_train, y_train)
    # Оценка точности классификации
    y_pred = cognitron.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    return accuracy

# Генетический алгоритм для обучения когнитрона
def ga_cognitron(population_size, max_generations, mutation_rate):
    # Инициализация популяции
    population = [Cognitron(random_weights=True) for _ in range(population_size)]

    # Цикл генетического алгоритма
    for generation in range(max_generations):
        # Оценка каждого когнитрона
        fitness_values = [evaluate_cognitron(cognitron, X_train, y_train) for cognitron in population]

        # Отбор лучших когнитронов
        elite = sorted(zip(fitness_values, population), reverse=True)[:population_size//2]
        population = [pair[1] for pair in elite]

        # Мутация и скрещивание
        for _ in range(population_size - len(elite)):
            parent1, parent2 = random.choice(population), random.choice(population)
            child = Cognitron(random_weights=True)
            child.weights = parent1.weights + parent2.weights
            child.thresholds = parent1.thresholds + parent2.thresholds
            if random.random() < mutation_rate:
                child.mutate()
            population.append(child)

    return population

# Запуск генетического алгоритма
best_cognitron = ga_cognitron(100, 1000, 0.1)

import numpy as np
import math
import random

generation_size = 30  # Jumlah Generasi
population_size = 10  # Jumlah Individu dalam satu populasi
kromosom_size = 8  # Panjang Kromosom -> Menggunakan Representasi biner
crossover_rate = 0.8  # Probabilitas crossover
mutation_rate = 0.1  # Probabilitas mutasi

min_X1 = -1  # Nilai minimum x1
max_X1 = 2  # Nilai maximum x1
min_X2 = -1  # Nilai minimum x2
max_X2 = 1  # Nilai maximum x2


def generate_Population():
    new_population = []
    for i in range(population_size):
        newGen = []
        for j in range(kromosom_size):
            randGen = random.randint(0, 1)
            newGen.append(randGen)
        new_population.append(newGen)
    return new_population


def decode_Kromosom(rbX1, raX1, rbX2, raX2, individu):
    x1 = rbX1 + (raX1 - rbX1)/(2**-1 + 2**-2 + 2**-3 + 2**-4) * \
        (individu[0]*2**-1 + individu[1]*2**-2 +
         individu[2]*2**-3 + individu[3]*2**-4)
    x2 = rbX2 + (raX2 - rbX2)/(2**-1 + 2**-2 + 2**-3 + 2**-4) * \
        (individu[4]*2**-1 + individu[5]*2**-2 +
         individu[6]*2**-3 + individu[7]*2**-4)
    return x1, x2


def individual_Fitness(x1, x2):
    h = math.cos(x1)*math.sin(x2) - x1/(x2**2+1)
    c = 4
    return c**-h


def evaluate_fitness(population):
    fitness = []
    for i in range(population_size):
        x1, x2 = decode_Kromosom(min_X1, max_X1, min_X2, max_X2, population[i])
        indv_fitness = individual_Fitness(x1, x2)
        fitness.append(indv_fitness)
    return fitness


def elitsm(population, fitness):
    pop = list(population)
    new_population = []
    first_best = np.argmax(fitness)
    new_population.append(pop[first_best])
    new_population.append(pop[first_best])
    return new_population


def individual_probability(fitness):
    total = np.sum(fitness)
    fitness_prob = []
    for i in range(len(fitness)):
        fit = fitness[i]/total
        fitness_prob.append(fit)
    return fitness_prob


def cumulative_probability(fitness):
    fitness_prob = individual_probability(fitness)
    cumulative_prob = []
    cumulative = 0
    for i in range(len(fitness)):
        cumulative += fitness_prob[i]
        cumulative_prob.append(cumulative)
    return cumulative_prob


def parent_selection(population, fitness):
    pop = list(population)
    cumulative_prob = cumulative_probability(fitness)
    roulette_rate = random.random()
    for i in range(population_size):
        if i > 0 and roulette_rate > cumulative_prob[i-1] and roulette_rate <= cumulative_prob[i]:
            parents = pop[i]
            break
        elif roulette_rate <= cumulative_prob[0]:
            parents = pop[i]
            break
    return parents


def crossover(male_parent, female_parent):
    rand_rate = random.random()
    offspring1 = list(male_parent)
    offspring2 = list(female_parent)
    if rand_rate <= crossover_rate:
        cross_point = random.randint(1, kromosom_size-1)
        offspring1 = male_parent[:cross_point] + female_parent[cross_point:]
        offspring2 = male_parent[:cross_point] + female_parent[cross_point:]
    return offspring1, offspring2


def mutate(offspring):
    for i in range(kromosom_size):
        randomRate = random.random()
        if randomRate <= mutation_rate:
            if offspring[i] == 1:
                offspring[i] = 0
            else:
                offspring[i] = 1
    return offspring


def generational_replacement():
    population = generate_Population()
    for i in range(generation_size):
        new_population = []
        fitness = []
        fitness = evaluate_fitness(population)
        new_population = elitsm(population, fitness)
        while len(new_population) < population_size:
            male_parent = parent_selection(population, fitness)
            female_parent = parent_selection(population, fitness)
            offspring1, offspring2 = crossover(male_parent, female_parent)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)
        population = new_population
    return population


# Main Program
population = generational_replacement()
fitness = evaluate_fitness(population)
best_individiual = np.argmax(fitness)
x1, x2 = decode_Kromosom(min_X1, max_X1, min_X2, max_X2, population[best_individiual])

print("Kromosom Terbaik = ", population[best_individiual])
print("X1 = ", x1)
print("X2 = ", x2)
print("Nilai Minimum = ", math.cos(x1)*math.sin(x2) - x1/(x2**2+1))

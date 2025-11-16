import random
import string
from dataclasses import dataclass
from typing import List, Tuple


# CONFIGURAÇÕES DO PROBLEMA
TARGET = "HELLO WORLD"   # frase que queremos evoluir
POP_SIZE = 200           # tamanho da população
TOURNAMENT_K = 5         # tamanho do torneio na seleção
MUTATION_RATE = 0.01     # probabilidade de mutação por gene (1%)
ELITISM = True           # manter o melhor indivíduo da geração
MAX_GENERATIONS = 14   # limite de gerações para evitar loop infinito

# Alfabeto permitido (letras maiúsculas + espaço)
ALPHABET = string.ascii_uppercase + " "


# ESTRUTURAS BÁSICAS
@dataclass
class Individual:
    genes: List[str]
    fitness: int = 0  # número de posições iguais ao TARGET

    def as_string(self) -> str:
        return "".join(self.genes)


# FUNÇÕES DO AG


def random_individual(length: int) -> Individual:
    """Cria um indivíduo aleatório (cromossomo = lista de caracteres)."""
    genes = [random.choice(ALPHABET) for _ in range(length)]
    return Individual(genes=genes, fitness=0)


def evaluate_fitness(ind: Individual, target: str) -> int:
    """Conta quantos caracteres batem exatamente com o alvo (fitness simples e rápido)."""
    score = sum(g == t for g, t in zip(ind.genes, target))
    ind.fitness = score
    return score


def tournament_selection(population: List[Individual], k: int) -> Individual:
    """
    Seleção por torneio:
    - amostra k indivíduos aleatórios
    - retorna o melhor (maior fitness)
    """
    contestants = random.sample(population, k)
    return max(contestants, key=lambda ind: ind.fitness)


def single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Crossover de ponto único:
    - escolhe um ponto de corte
    - filho1 = (genes do pai1 até corte) + (genes do pai2 após corte)
    - filho2 = (genes do pai2 até corte) + (genes do pai1 após corte)
    """
    length = len(parent1.genes)
    if length <= 1:
        # Sem sentido cruzar genes de tamanho 1
        return Individual(parent1.genes.copy()), Individual(parent2.genes.copy())

    cut = random.randint(1, length - 1)
    child1_genes = parent1.genes[:cut] + parent2.genes[cut:]
    child2_genes = parent2.genes[:cut] + parent1.genes[cut:]
    return Individual(child1_genes), Individual(child2_genes)


def mutate(ind: Individual, mutation_rate: float) -> None:
    """
    Mutação ponto-a-ponto:
    - para cada gene, com pequena probabilidade, troca por um caractere aleatório
    """
    for i in range(len(ind.genes)):
        if random.random() < mutation_rate:
            ind.genes[i] = random.choice(ALPHABET)


def make_initial_population(size: int, length: int) -> List[Individual]:
    """Cria a população inicial completamente aleatória."""
    return [random_individual(length) for _ in range(size)]


# LOOP EVOLUTIVO


def genetic_algorithm():
    """Algoritmo Genético simples para evoluir uma string alvo a partir de uma população aleatória.
    O algoritmo utiliza seleção por torneio, crossover de ponto único e mutação ponto-a-ponto
    em uma população de indivíduos representados como listas de caracteres."""
    target_len = len(TARGET)
    population = make_initial_population(POP_SIZE, target_len)
    # Avalia população inicial
    for ind in population:
        evaluate_fitness(ind, TARGET)

    best = max(population, key=lambda i: i.fitness)

    generation = 0
    while generation < MAX_GENERATIONS and best.fitness < target_len:
        new_population = []

        # Elitismo: carrega o melhor indivíduo para a próxima geração
        if ELITISM:
            new_population.append(Individual(best.genes.copy(), best.fitness))

        # Gera novos indivíduos até completar a população
        while len(new_population) < POP_SIZE:
            # Seleção: escolhe dois pais via torneio
            parent1 = tournament_selection(population, TOURNAMENT_K)
            parent2 = tournament_selection(population, TOURNAMENT_K)

            # Cruzamento
            child1, child2 = single_point_crossover(parent1, parent2)

            # Mutação
            mutate(child1, MUTATION_RATE)
            mutate(child2, MUTATION_RATE)

            # Avalia os filhos
            evaluate_fitness(child1, TARGET)
            evaluate_fitness(child2, TARGET)

            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population
        best = max(population, key=lambda i: i.fitness)
        generation += 1

        # Imprime progresso a cada 5 gerações ou se encontrar a solução
        if generation % 5 == 0 or best.fitness == target_len:
            print(
                f"Geração {generation:4d} | Melhor fitness: {best.fitness:2d} | '{best.as_string()}'")

    # Resultado final
    print("\n=== RESULTADO ===")
    print(f"Gerações: {generation}")
    print(
        f"Melhor indivíduo: '{best.as_string()}' (fitness {best.fitness}/{target_len})")
    return best, generation


genetic_algorithm()

import numpy as np
import copy
import Problem
import CrossoverFunction
import random
from operator import itemgetter
import math
import matplotlib.pyplot as plt


def randomSolution(points, path_length, taxi=(35.203958315434434, 31.788860911945207)):
    solution = np.zeros((2 * path_length + 1, 4))
    solution[0] = [taxi[0], taxi[1], -1, -1]
    blank_spot = [x for x in range(1, 2 * path_length + 1)]
    for i in range(path_length):
        two_rand_spot = random.sample(blank_spot, 2)
        a, b = min(two_rand_spot), max(two_rand_spot)
        solution[a][:2] = points[i][:2]
        solution[a][2] = i
        solution[a][3] = 0
        solution[b][:2] = points[i][2:4]
        solution[b][2] = i
        solution[b][3] = 1
        blank_spot.remove(a)
        blank_spot.remove(b)
    return solution


def Crossover(path1, path2, r):
    path1, path2 = CrossoverFunction.crossover(path1, path2, r)
    return [path1, path2]

# should the mutation rate be constant or reduce after generations
def mutate(path, r):
    x = random.choices([0, 1], [1 - r, r])
    if x == 0:
        return path
    path_len = path.shape[0]
    MUTATION_NUMBER = 2
    mutation_name_1, mutation_name_2 = random.sample(range(path_len // 2), MUTATION_NUMBER)
    start_1, end_1 = find_by_name(path, mutation_name_1)
    start_2, end_2 = find_by_name(path, mutation_name_2)
    old_start, old_end = copy.deepcopy(path[start_1]), copy.deepcopy(path[end_1])
    path[start_1], path[end_1] = path[start_2], path[end_2]
    path[start_2], path[end_2] = old_start, old_end
    return path

def find_by_name(path, name):
    start, end = -1, -1
    for i in range(path.shape[0]):
        if path[i][2] == name:
            if path[i][3] == 0:
                start = i
            if path[i][3] == 1:
                end = i
                break
    return start, end

# get a path and return the distance of it
def evaluation(path):
    length = path.shape[0]
    total_len = 0
    for i in range(1, length):
        total_len += Problem.d(path[i - 1][:2], path[i][:2])
    return total_len


# gets the scores of each path and returning what will be his precntge in the genretion
def precentge(l, n):
    s = sum(l) / (2 * len(l))
    s = [(math.exp((-(sc - min(l)) / n))) for sc in l]
    return [float(i) / sum(s) for i in s]


# gets poplation size, Crossover rate, mutate rate, Number of genrations
def GA(points, taxi, Population_size, Crossover_rate, Mutation_rate, Generation_number, type=0):
    n = points.shape[0]
    Poplation = [randomSolution(points, n, taxi) for _ in range(Population_size)]
    Bestdistace = []
    bestpath = []
    for cur_gen_num in range(Generation_number):
        scores = [evaluation(path) for path in Poplation]
        index, element = min(enumerate(scores), key=itemgetter(1))
        bestpath.append(Poplation[index])
        Bestdistace.append(element)

        pscores = precentge(scores, n)
        # croos over
        perents = [random.choices(Poplation, pscores, k=2) for _ in range(int(Population_size / 2))]
        Poplation = []
        for pair in perents:
            Poplation.extend(Crossover(pair[0], pair[1], Crossover_rate))

        # mutate
        for path in Poplation:
            # mutate_r = Mutation_rate / (cur_gen_num + 1)
            path = mutate(path, Mutation_rate)
        if cur_gen_num == Generation_number // 4:
            print("25%")
        if cur_gen_num == Generation_number // 2:
            print("50%")
        if cur_gen_num == 3 * Generation_number // 4:
            print("75%")
    scores = [evaluation(p) for p in Poplation]
    index, element = min(enumerate(scores), key=itemgetter(1))
    bestpath.append(Poplation[index])
    Bestdistace.append(element)
    plt.plot(range(len(Bestdistace)), Bestdistace)
    # plt.show()
    index, element = min(enumerate(Bestdistace), key=itemgetter(1))
    theBest = bestpath[index]

    return theBest[:, :2]

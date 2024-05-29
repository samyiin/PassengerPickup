import sys
import os
import re
from problem_for_local_search import *

algos_list = {"G": GENETIC, "HC": HILL_CLIMBING, "SA": SIMULATED_ANNEALING,
              "FC": FIRST_CHOICE, "RR": RESTART_HILL, "B": BEAM, "SB": STOCHASTIC_BEAM, "GREEDY": GREEDY, "T": TABU,
              "LA": LATE_ACCEPTANCE}

methods_list = {"SI": SWAP_INDEXES, "SA": SWAP_ADJACENT, "IAS": INSERT_AND_SHIFT}
RANDOM = "-r"
TWO_FLOAT_PATTERN = "([0-9]*[.])?[0-9]+ ([0-9]*[.])?[0-9]+[ ]?"
FOUR_FLOAT_PATTERN = "([0-9]*[.])?[0-9]+ ([0-9]*[.])?[0-9]+ ([0-9]*[.])?[0-9]+ ([0-9]*[.])?[0-9]+[ ]?"
LONGITUDE_MAX = 35.2487
LONGITUDE_MIN = 35.1694
LATITUDE_MAX = 31.8062
LATITUDE_MIN = 31.7493

POPULATION_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
CROSSOVER_TYPE = 0
GENERATION_NUMBER = 10


def check_file(file_path):
    file = open(file_path, 'r')
    line_1 = file.readline()
    if not line_1:
        return False
    if re.match(TWO_FLOAT_PATTERN, line_1) is None:
        return False
    line_1_temp = line_1.split(" ")
    if not (LONGITUDE_MIN <= float(line_1_temp[0]) <= LONGITUDE_MAX) \
            or not (LATITUDE_MIN <= float(line_1_temp[1]) <= LATITUDE_MAX):
        return False
    count = 0
    while True:
        # Get next line from file
        line = file.readline()
        if not line:
            if count == 0:
                return False
            break
        count += 1
        if re.match(FOUR_FLOAT_PATTERN, line) is None:
            return False
        line_temp = line.split(" ")
        if not (LONGITUDE_MIN <= float(line_temp[0]) <= LONGITUDE_MAX) \
                or not (LATITUDE_MIN <= float(line_temp[1]) <= LATITUDE_MAX) \
                or not (LONGITUDE_MIN <= float(line_temp[2]) <= LONGITUDE_MAX) \
                or not (LATITUDE_MIN <= float(line_temp[3]) <= LATITUDE_MAX):
            return False
    return True


def get_info(file_path):
    file = open(file_path, 'r')
    carrier_temp = file.readline()
    carrier_coordinate = carrier_temp.split(" ")
    for i in range(len(carrier_coordinate)):
        carrier_coordinate[i] = float(carrier_coordinate[i])
    carrier_location = tuple(carrier_coordinate)
    passanger_mtx = []
    while True:
        # Get next line from file
        line = file.readline()
        if not line:
            break
        temp = line.split(" ")
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        passanger_mtx.append(temp)
    items = np.array(passanger_mtx)
    return items, carrier_location


def main(argv):
    if len(argv) == 1:
        return run(range_of_passengers=[5], capacity=INF, algos={HILL_CLIMBING: [INSERT_AND_SHIFT]})

    if argv[1] not in [INF, ONE]:
        print("choose problem type: 1 or 'inf'")
        return
    else:
        capacity = argv[1]

    if int(argv[2]) <= 0:
        print("choose number of passengers")
        return
    else:
        number_of_passengers = int(argv[2])

    if argv[3] != RANDOM:
        file_path = argv[3]
        if not os.path.isfile(file_path):
            print("File does not exist")
            return
        if not check_file(file_path):
            print("This file has incorrect information\n")
            return
        else:
            points, start_point = get_info(file_path)
    else:
        points, start_point = None, None

    if argv[4] not in algos_list:
        print("choose algorithm from:", algos_list)
        return
    else:
        algo = algos_list[argv[4]]
        if algo == GREEDY:
            return run(range_of_passengers=[number_of_passengers], capacity=capacity, points=points,
                       start_point=start_point)

    if algo == GENETIC:
        if len(argv) == 5:
            method = [[POPULATION_SIZE, CROSSOVER_RATE, MUTATION_RATE, GENERATION_NUMBER,
                      CROSSOVER_TYPE]]
        else:
            if capacity == ONE:
                method = [argv[5:10]]
            else:
                method = [argv[5:9]]

    else:
        if argv[5] not in methods_list:
            print("choose successors method from:", methods_list)
            return
        else:
            method = [methods_list[argv[5]]]

    return run(range_of_passengers=[number_of_passengers], capacity=capacity, points=points, start_point=start_point,
               algos={algo: method})


if __name__ == "__main__":
    main(sys.argv)

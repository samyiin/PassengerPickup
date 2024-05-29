USER GUIDE:
option_1:
run passengers_pickup.py

option_2:
run passengers_pickup.py from cmd, with the following parameters:
<capacity> <num_of_passengers> <mode: random or file_path> <algorithm> <method or params>

capacity = ["1", "inf"]
num_of_passengers = positive integer
mode = "-r" for random problem, file_path for import problem from file.
algorithm = {"G": GENETIC, "HC": HILL_CLIMBING, "SA": SIMULATED_ANNEALING,
              "FC": FIRST_CHOICE, "RR": RESTART_HILL, "B": BEAM, "SB": STOCHASTIC_BEAM, "GREEDY": GREEDY, "T": TABU,
              "LA": LATE_ACCEPTANCE}
if algorithm == "GREEDY" -> just run
if algorithm == "GENETIC" -> params = list of 5 parameters. if not given -> default parameters.
else -> method = {"SI": SWAP_INDEXES, "SA": SWAP_ADJACENT, "IAS": INSERT_AND_SHIFT}

*****
-file format-:
text file, with num_of_passengers+1 lines.
the first line contains 2 numbers:
the first between LONGITUDE_MIN and LONGITUDE_MAX, and the second between LATITUDE_MIN and LATITUDE_MAX
the rest of num_of_passengers lines contain 4 numbers, each pair must be in the same range like the first line.

LONGITUDE_MAX = 35.2487
LONGITUDE_MIN = 35.1694
LATITUDE_MAX = 31.8062
LATITUDE_MIN = 31.7493
*****
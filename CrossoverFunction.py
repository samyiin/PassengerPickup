import random
import numpy as np

'''
only partial fail_safe here, I didn't wrote a full check for if parent 1, parent 2 is legal, because it's time consuming
, and (hopefully) if it's checked elsewhere, then delete "parent_legal" function
the initial val for child for all coordinate is [0000], so don't test this instance
I could have change output for all crossover to two children, but it's clearer and more flexible (yet more time 
consuming) to do it twice:
crossover(P_1, P_2), crossover(P_2, P_1)
the default crossover method I set is ordered crossover, feel free to try out:
    cycle_crossover
    partially_mapped_crossover
there are some other crossover method's, but they only worked for adjacency representation (a mapping of one node and 
it's next):
    alternating_edge_crossover (careful)
    Heuristic_crossover
    
'''


def crossover(parent_1, parent_2, crossover_rate,type=0):
    if crossover_rate > 1:
        raise Exception("crossover rate should be in range [0, 1]")
    if not parents_legal(parent_1, parent_2):
        raise Exception("the two parent's should contain same set of elements")
    # may not crossover
    if random.choices([0, 1], [1 - crossover_rate, crossover_rate]) == 0:
        return parent_1, parent_2
    shape = parent_1.shape
    num_rows, num_cols = shape[0], shape[1]
    # below are changeable
    if type==0:
        child_1 = order_crossover(parent_1, parent_2, num_rows, num_cols)
        child_2 = order_crossover(parent_2, parent_1, num_rows, num_cols)
    elif type==1:
        child_1 = cycle_crossover(parent_1, parent_2, num_rows, num_cols)
        child_2 = cycle_crossover(parent_2, parent_1, num_rows, num_cols)
    elif type==2:
        child_1 = partially_mapped_crossover(parent_1, parent_2, num_rows, num_cols)
        child_2 = partially_mapped_crossover(parent_2, parent_1, num_rows, num_cols)
    else:
        child_1 = order_crossover(parent_1, parent_2, num_rows, num_cols)
        child_2 = order_crossover(parent_2, parent_1, num_rows, num_cols)
    return child_1, child_2


# cycle crossover
# for one passenger, correctness for infinity passenger unknown
# input: parent 1, parent 2, number of rows, and number of columns
# output: one child
def cycle_crossover(parent_1, parent_2, num_rows, num_cols):
    counter = 0
    child = np.zeros((num_rows, num_cols))
    visited = []
    while True:
        index = find_next_index(visited)
        if index == num_rows:
            break
        box = []
        while True:
            if index in box:
                break
            box.append(index)
            visited.append(index)
            index = find_index_same(parent_2[index], parent_1)
        if counter % 2 == 0:
            for idx in box:
                child[idx] = parent_1[idx]
        else:
            for idx in box:
                child[idx] = parent_2[idx]
        counter += 1
    return child


# partially crossover
# for one passenger, correctness for infinity passenger unknown
# input: parent 1, parent 2, number of rows, and number of columns
# output: one child
def partially_mapped_crossover(parent_1, parent_2, num_rows, num_cols):
    child = np.zeros((num_rows, num_cols))
    visited = []
    for i in range(num_rows // 2):
        visited.append(i)
        child[i] = parent_1[i]
        if find_index_same(parent_2[i], parent_1) >= num_rows // 2:
            idx = find_index_same(parent_1[i], parent_2)
            while idx < num_rows //2:
                idx = find_index_same(parent_1[idx], parent_2)
            else:
                child[idx] = parent_2[i]
                visited.append(idx)
    for i in range(num_rows):
        if i not in visited:
            child[i] = parent_2[i]
    return child



    return child


# partially crossover
# for one or infinity passenger
# input: parent 1, parent 2, number of rows, and number of columns
# output: one child
def order_crossover(parent_1, parent_2, num_rows, num_cols):
    child = np.zeros((num_rows, num_cols))
    visited = []
    for i in range(num_rows // 2):
        child[i] = parent_1[i]
        visited.append(parent_1[i])
    idx = num_rows // 2
    for i in range(num_rows):
        if not_in(parent_2[i], visited):
            child[idx] = parent_2[i]
            idx += 1
        if idx >= num_rows:
            break
    return child


# input a row in parent 1 (a coordinate, or 2 coordinates)
# assume: exist same row as parent_1_row in parent_2
# output: its index in parent 2
def find_index_same(parent_1_row, parent_2):
    counter = 0
    for row in parent_2:
        if np.array_equal(row, parent_1_row):
            return counter
        counter += 1
    return -1


# auxiliary function, not important
def find_next_index(visited):
    index = 0
    while True:
        if index in visited:
            index += 1
        else:
            return index


# check if parent is legal, true if legal
def parents_legal(parent_1, parent_2):
    return parent_1.shape == parent_2.shape



# input : array_1: and an array, array_2: an array of arrays
# output: array_1 doesn't exist in array_2
# major loophole: if array_2 have equal elements
def not_in(array_1, array_2):
    for i in array_2:
        if np.array_equal(i, array_1):
            return False
    return True

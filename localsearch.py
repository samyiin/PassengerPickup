import math
from utils import *
from collections import deque

import numpy as np

MAX_ITERATIONS_NUM = 10000

SWAP_INDEXES = 'swap_indexes'
INSERT_AND_SHIFT = 'insert_and_shift'
SWAP_ADJACENT = 'swap_adjacent'
# np.random.seed(1)

def print_state(state, show=True):
    text = ""
    for point in state:
        if point.is_src_point():
            text += "S"
            if show:
                print("S" + str(point), " -> ", end="")
                # print("min:", str(point.min_ind)+",", "max:", point.max_ind, end=", ")

        if point.is_dest_point():
            text += "D"
            if show:
                print("D" + str(point), " -> ", end="")
                # print("min:", str(point.min_ind)+",", "max:", point.max_ind, end=", ")

        text += str(point) + " -> "
    if show:
        print("END\n")
    text += "END"
    return text

def copy_state(state):
    new_state = []
    for point in state:
        new_state.append(point.__deepcopy__())
    return new_state


def flipCoin(p):
    r = np.random.random()
    return r < p


class SearchProblem:
    """
    This class outlines thxe structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        return []

    def get_start_point(self):
        return Point()

    def get_successors(self, state, method=SWAP_INDEXES, max_successors=None, random=False, generate_all=False):
        return []

    def get_random_successor(self, state):
        return []

    def get_state_cost(self, state, start_point=None):
        return 1

    def get_random_states(self, k=1):
        return []



def hill_climbing_search(problem: SearchProblem, initial_state=None, epsilon=0.0, successors_method=INSERT_AND_SHIFT):
    """
    problem - the search problem.
    epsilon - the convergence parameter
    returns a local minima state.
    """
    best = initial_state if initial_state else problem.get_start_state()
    best_cost = problem.get_state_cost(best)
    idle_iterations = 0
    sols = [best]
    for i in range(MAX_ITERATIONS_NUM):
        # print_state(best)

        if idle_iterations > epsilon * i:
            break
        children = problem.get_successors(best, method=successors_method)
        if children is None:
            break
        next = min(children, key=lambda state: problem.get_state_cost(state))

        next_cost = problem.get_state_cost(next)
        if next_cost <= best_cost:
            idle_iterations = idle_iterations + 1 if next_cost == best_cost else 0
            best = next
            best_cost = next_cost
            sols.append(best)

        else:
            break
    return sols


def greedy_search(problem: SearchProblem, capacity=math.inf):
    sol = []
    closest_point = None
    min_dist = math.inf
    closest_point_ind = 0
    points = copy_state(problem.get_start_state())

    if capacity == 1:
        for i in range(len(points) // 2):
            point = points[2*i]
            cur_dist = point.distance(problem.get_start_point())
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_point = points[2*i:2*i+2]
                closest_point_ind = i
        sol += closest_point
        points[2*closest_point_ind] = None

        for j in range(1, len(points) // 2):
            closest_point = None
            min_dist = math.inf
            closest_point_ind = 0
            for i in range(len(points) // 2):
                point = points[2*i]
                if point is None:
                    continue
                cur_dist = point.distance(sol[2*j - 1])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_point = points[2*i:2*i+2]
                    closest_point_ind = i

            if closest_point is not None:
                points[2*closest_point_ind] = None
                sol += closest_point

    else:
        for i in range(len(points)):
            point = points[i]
            cur_dist = point.distance(problem.get_start_point())
            if point.is_src_point() and cur_dist < min_dist:
                min_dist = cur_dist
                closest_point = point
                closest_point_ind = i
        sol.append(closest_point)
        points[closest_point_ind] = 0

        for j in range(1, len(points)):
            closest_point = None
            min_dist = math.inf
            closest_point_ind = 0
            for i in range(len(points)):
                point = points[i]
                if type(point) == int:
                    continue
                if point.is_dest_point() and type(points[point.min_ind - 1]) != int:
                    continue
                cur_dist = point.distance(sol[j - 1])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_point = point
                    closest_point_ind = i

            if closest_point is not None:
                points[closest_point_ind] = j
                if closest_point.is_dest_point():

                    closest_point.min_ind = points[closest_point.min_ind - 1] + 1
                    sol[closest_point.min_ind - 1].max_ind = j - 1

                sol.append(closest_point)

    return sol


def simulated_annealing_search(problem: SearchProblem, initial_state=None, successors_method=SWAP_INDEXES, initial_temp=100000, epsilon=0.01):
    """
    a hill climbing variation, with 2 differences: 1) allows worsening moves with probability which is dependant on
    the time passed since start of execution and on the worsening level. 2) selects successor randomly, and doesn't
    pick the best successor (necessarily).
    """
    best = initial_state if initial_state else problem.get_start_state()
    best_cost = problem.get_state_cost(best)
    for i in range(MAX_ITERATIONS_NUM):
        temperature = initial_temp / (i+1)
        # alternative - geometric cooling: temperature = alpha * temperature
        if temperature < epsilon:
            return best
        # alternative - get all successors, and flip a coin to decide if to choose the best one or a random one:
        # children = problem.get_successors(best)
        # next = min(children, key=lambda state: problem.get_state_cost(state))
        # next_cost = problem.get_state_cost(next)
        # cost_diff = next_cost - best_cost
        # if flipCoin(math.exp(cost_diff / temperature)):
        #     next = children[np.random.choice(len(children))]
        #     next_cost = problem.get_state_cost(next)
        # best = next
        # best_cost = next_cost
        # print_state(best)
        ############
        next_state = problem.get_successors(best, random=True, method=successors_method)
        next_cost = problem.get_state_cost(next_state)
        cost_diff = next_cost - best_cost
        if cost_diff <= 0 or flipCoin(math.exp(-cost_diff / temperature)):
            best = next_state
            best_cost = next_cost
    return best


def local_beam_search(problem: SearchProblem, successors_method=SWAP_INDEXES, k=3, epsilon=0.01):
    """
    a hill climbing variation. keeps track of k states rather than just one. at each iteration, the k best successors
    of the current k states (combined) are generated, and they serve as "current states" for the next iteration, if
    they indeed offer some improvement.
    """
    best_states = problem.get_random_states(k)
    best_states.sort(key=lambda state: problem.get_state_cost(state))
    best_cost = problem.get_state_cost(best_states[0])
    idle_iterations = 0
    sols = [best_states[0]]
    for i in range(MAX_ITERATIONS_NUM):
        # print(i)
        if idle_iterations > epsilon * i:
            break
        successors = []
        for j in range(len(best_states)):
            cur_successors = problem.get_successors(best_states[j], method=successors_method, generate_all=True)
            # cur_successors.sort(key=lambda state: problem.get_state_cost(state))
            # cur_successors = cur_successors[:min(len(cur_successors), k)]
            # if not (len(successors) != 0 and cur_successors[0] >= successors[-1]):
            successors.extend(cur_successors)
        successors.sort(key=lambda state: problem.get_state_cost(state))
        successors_best_cost = problem.get_state_cost(successors[0])
        if successors_best_cost <= best_cost:
            idle_iterations = idle_iterations + 1 if successors_best_cost == best_cost else 0
            closed = set()
            count = 0
            best_states = []
            for succ in successors:
                if tuple(succ) not in closed:
                    best_states.append(succ)
                    closed.add(tuple(succ))
                    count += 1
                    if count == k:
                        break
            best_cost = successors_best_cost
            sols.append(best_states[0])
        else:
            break
    return sols


def stochastic_local_beam_search(problem: SearchProblem, successors_method=SWAP_INDEXES, k=3, epsilon=0.01):
    """
    same as local beam search, with one difference: instead of choosing the best k successors, the k successors are
    selected randomly.
    """
    best_states = problem.get_random_states(k)
    best_cost = problem.get_state_cost(best_states[0])
    idle_iterations = 0
    for i in range(MAX_ITERATIONS_NUM):
        if idle_iterations > epsilon * i:
            break
        successors = []
        for j in range(k):
            successors.extend(problem.get_successors(best_states[j], method=successors_method, generate_all=True))
        # select k random successors
        probs = np.zeros(len(successors))
        for l in range(len(probs)):
            probs[l] = np.exp(1 / problem.get_state_cost(successors[l]))
        probs /= np.sum(probs)
        random_indexes = np.random.choice(len(successors), min(k, len(successors)), replace=False, p=probs)
        successors = [successors[index] for index in random_indexes]
        successors_best_cost = min([problem.get_state_cost(state) for state in successors])
        if successors_best_cost <= best_cost:
            idle_iterations = idle_iterations + 1 if successors_best_cost == best_cost else 0
            best_states = successors
            best_cost = successors_best_cost
        else:  # TODO is it necessary? maybe we should allow worsening moves?
            break
    return min(best_states, key=lambda state: problem.get_state_cost(state))


def random_restart_hill_climbing(problem: SearchProblem, successors_method=SWAP_INDEXES, k=3):
    """
    runs several hill climbing searches and then selects the best result among them
    """
    solutions = []
    for i in range(k):
        initial_state = problem.get_random_states()
        solutions.append(hill_climbing_search(problem, initial_state, successors_method=successors_method)[-1])
    return min(solutions, key=lambda state: problem.get_state_cost(state))


def first_choice_hill_climbing(problem: SearchProblem, initial_state=None, successors_method=SWAP_INDEXES, epsilon=0.01):
    """
    A hill climbing variation. instead of choosing the best among all successor, the algorithm selects repeatedly a
    random successor, and continues with it if it is better than the current best solution. This approach should be
    useful when there are many successors for each state.
    """
    best = initial_state if initial_state else problem.get_start_state()
    best_cost = problem.get_state_cost(best)
    idle_iterations = 0
    for i in range(MAX_ITERATIONS_NUM):
        if idle_iterations > epsilon * i:
            break
        counter = 0
        while True:
            if counter >= MAX_ITERATIONS_NUM:
                return best
            next = problem.get_successors(best, random=True, method=successors_method)
            next_cost = problem.get_state_cost(next)
            if next_cost <= best_cost:
                idle_iterations = idle_iterations + 1 if next_cost == best_cost else 0
                best = next
                best_cost = next_cost
                break
            counter += 1

    return best


def late_acceptance_hill_climbing(problem: SearchProblem, L=3, initial_state=None, successors_method=SWAP_INDEXES, epsilon=0.02):
    """
    A hill climbing variation, with one major difference: instead of only comparing the best neighbor with the quality
    of the current solution, we also compare it to the solution L iterations ago - thus allowing worsening moves, if
    they improve upon the best move some iterations ago.
    """
    best = initial_state if initial_state else problem.get_start_state()
    best_cost = problem.get_state_cost(best)
    recent_states_cost = [best_cost] * L
    idle_iterations = 0
    for i in range(MAX_ITERATIONS_NUM):
        if idle_iterations > epsilon * i:
            break
        children = problem.get_successors(best, method=successors_method)
        if children is None:
            break
        best_neighbor = children[0]
        best_neighbor_cost = problem.get_state_cost(best_neighbor)
        idle_iterations = idle_iterations + 1 if best_neighbor_cost >= best_cost else 0
        v = i % L  # the last_currents is actually a circular list
        if best_neighbor_cost <= best_cost or best_neighbor_cost <= recent_states_cost[v]:
            best = best_neighbor
            best_cost = best_neighbor_cost
            recent_states_cost[v] = best_cost
    return best


def tabu_search(problem: SearchProblem, max_tabu_size, initial_state=None, successors_method=SWAP_INDEXES, epsilon=0.02):
    """
    A hill climbing variation. At every step, the algorithm selects the best valid neighbor of the current state to be
    the next current state. A neighbor will be considered "valid" if it is not currently in the maintained Tabu list -
    a list that contains previous states which were chosen to be "current state" in recent iterations. The algorithm
    chooses the best valid neighbor even when its cost is higher then the current best found solution, and thus avoids
    the local minima problem.
    """
    result_state = initial_state if initial_state else problem.get_start_state()
    result_cost = problem.get_state_cost(result_state)
    best_candidate = result_state
    tabu_list = deque()
    tabu_list.append(result_state)
    idle_iterations = 0
    for i in range(MAX_ITERATIONS_NUM):
        if idle_iterations > epsilon * i:
            break
        children = problem.get_successors(best_candidate, method=successors_method, generate_all=True)
        best_candidate = children[0]
        best_candidate_cost = problem.get_state_cost(best_candidate)
        for candidate in children:  # selects best valid candidate among the neighbors of the current state
            if candidate not in tabu_list and problem.get_state_cost(candidate) < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = problem.get_state_cost(best_candidate)
        idle_iterations = idle_iterations + 1 if best_candidate_cost >= result_cost else 0
        if best_candidate_cost <= result_cost:
            result_state = best_candidate
            result_cost = best_candidate_cost
        tabu_list.append(best_candidate)
        if len(tabu_list) > max_tabu_size:
            tabu_list.popleft()
    return result_state

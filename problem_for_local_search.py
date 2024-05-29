import matplotlib.pyplot as plt
import matplotlib
from Problem import crateRandomProblem
import inf
from GenticAlgoritem import GA
from localsearch import *
import time

ONE = "1"
INF = "inf"

HILL_CLIMBING = "hill climbing"
FIRST_CHOICE = "first choice"
RESTART_HILL = "random restart hill climbing"
LATE_ACCEPTANCE = "late acceptance"
TABU = "tabu search"
BEAM = "beam search"
STOCHASTIC_BEAM = "stochastic beam search"
SIMULATED_ANNEALING = "simulated annealing "
GREEDY = "greedy"
GENETIC = "genetic"
BRUTE = "brute force"

START = (35.203958315434434, 31.788860911945207)
matplotlib.use('Qt5Agg')
Box = (35.1694, 35.2487,
       31.7493, 31.8062)
methods = {INSERT_AND_SHIFT, SWAP_ADJACENT, SWAP_ADJACENT}
all_algos = {GENETIC, SIMULATED_ANNEALING, HILL_CLIMBING, BEAM, GREEDY, STOCHASTIC_BEAM, TABU, LATE_ACCEPTANCE,
             RESTART_HILL}
# NOT COMPLETE
max_range = 50


def import_from_points_array(points):
    if points.shape[1] == 4:
        num_of_passengers = len(points)
        new_list = []
        for i in range(num_of_passengers):
            p_s = Point(points[i][0], points[i][1])
            p_d = Point(points[i][2], points[i][3])
            new_list += [p_s, p_d]

            p_s.set_max_ind(2 * i)
            p_s.set_min_ind(0)

            p_d.set_max_ind(num_of_passengers * 2 - 1)
            p_d.set_min_ind(2 * i + 1)

            p_d.add_src(p_s)
            p_s.add_dest(p_d)
        return new_list

    elif points.shape[1] == 2:
        num_of_passengers = len(points) // 2
        new_list = []
        for i in range(num_of_passengers):
            p_s = Point(points[2 * i][0], points[2 * i][1])
            p_d = Point(points[2 * i + 1][0], points[2 * i + 1][1])
            new_list += [p_s, p_d]

            p_s.set_max_ind(2 * i)
            p_s.set_min_ind(0)

            p_d.set_max_ind(num_of_passengers * 2 - 1)
            p_d.set_min_ind(2 * i + 1)

            p_d.add_src(p_s)
            p_s.add_dest(p_d)
        return new_list


class InfCapacityProblem(SearchProblem):

    def __init__(self, num_of_passengers=10, start_point=None, points=None):

        if start_point is not None:
            if isinstance(start_point, Point):
                self.start_point = start_point
            else:
                self.start_point = Point(start_point[0], start_point[1])
        else:
            self.start_point = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))

        if points is not None:
            self.points = import_from_points_array(points)
        else:
            self.points = []
            for i in range(num_of_passengers):
                p_s = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))
                p_d = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))
                # p_s = Point(np.random.choice(max_range), np.random.choice(max_range))
                # p_d = Point(np.random.choice(max_range), np.random.choice(max_range))
                self.points += [p_s, p_d]

                p_s.set_max_ind(2 * i)
                p_s.set_min_ind(0)

                p_d.set_max_ind(num_of_passengers * 2 - 1)
                p_d.set_min_ind(2 * i + 1)

                p_d.add_src(p_s)
                p_s.add_dest(p_d)

    def get_start_state(self):
        return self.points

    def get_start_point(self):
        return self.start_point

    def _get_cost_without_change(self, state, i, j, method):
        n = len(state)
        if method == INSERT_AND_SHIFT:
            first_point = state[i]
            last_point = state[j]
            min_cost = math.inf
            if not (first_point.is_src_point() and (j > first_point.max_ind)):
                parts = [state[:i], state[i + 1:j + 1], [state[i]]]
                if j < n - 1:
                    parts.append(state[j + 1:])
                min_cost = self.get_state_cost(parts, parts=True)
            # return min_cost

            if not (last_point.is_dest_point() and i < last_point.min_ind):
                parts = [state[:i], [state[j]], state[i:j]]
                if j + 1 < n:
                    parts.append(state[j + 1:])
                cur_cost = self.get_state_cost(parts, parts=True)
                if cur_cost < min_cost:
                    min_cost = cur_cost
            return min_cost

        elif method == SWAP_ADJACENT:
            first_point = state[i]
            last_point = state[j]
            if first_point.is_src_point() and (j > first_point.max_ind):
                return math.inf
            if last_point.is_dest_point() and (i < last_point.min_ind):
                return math.inf
            parts = [state[:i], [state[j]], [state[i]]]
            if j + 1 < n:
                parts.append(state[j + 1:])
            return self.get_state_cost(parts, parts=True)
        else:
            first_point = state[i]
            last_point = state[j]
            if first_point.is_src_point() and j > first_point.max_ind:
                return -1
            if last_point.is_dest_point() and i < last_point.min_ind:
                return math.inf
            parts = [state[:i], [state[j]], state[i + 1:j], [state[i]]]
            if j + 1 < n:
                parts.append(state[j + 1:])
            return self.get_state_cost(parts, parts=True)

    def _swap_adjacent_successors_helper(self, state, i, j):
        if j >= len(state):
            return None
        first_point = state[i]
        last_point = state[j]
        # check if first_point can't move to the location of last_point, and vice versa:

        if first_point.is_src_point() and (j > first_point.max_ind):
            return None
        if last_point.is_dest_point() and (i < last_point.min_ind):
            return None

        new_state = copy_state(state)

        # update the index options of the relevant points:
        if first_point.is_src_point():
            new_state[first_point.max_ind + 1].min_ind += 1

        if first_point.is_dest_point():
            new_state[first_point.min_ind - 1].max_ind += 1

        if last_point.is_src_point():
            new_state[last_point.max_ind + 1].min_ind -= 1
        if last_point.is_dest_point():
            new_state[last_point.min_ind - 1].max_ind -= 1

        new_state[i], new_state[j] = new_state[j], new_state[i]

        return new_state

    def _insert_and_shift_successors_helper(self, state, i, j):
        first_point = state[i]
        last_point = state[j]
        successors = []
        if not (first_point.is_src_point() and (j > first_point.max_ind)):

            new_state = copy_state(state)

            # update the index options of the relevant points:
            trans_point = new_state[i]
            for k in range(i + 1, j + 1):
                cur_point = state[k]
                if cur_point.is_src_point():
                    new_state[cur_point.max_ind + 1].min_ind -= 1

                if cur_point.is_dest_point():
                    new_state[cur_point.min_ind - 1].max_ind -= 1

            if trans_point.is_src_point():
                new_state[trans_point.max_ind + 1].min_ind = j + 1
            if trans_point.is_dest_point():
                new_state[trans_point.min_ind - 1].max_ind = j - 1

            new_state[i:j] = new_state[i + 1:j + 1]
            new_state[j] = trans_point
            successors.append(new_state)

        if not (last_point.is_dest_point() and i < last_point.min_ind):
            new_state = copy_state(state)
            # update the index options of the relevant points:
            trans_point = new_state[j]
            for k in range(i, j):
                cur_point = state[k]
                if cur_point.is_src_point():
                    new_state[cur_point.max_ind + 1].min_ind += 1

                if cur_point.is_dest_point():
                    new_state[cur_point.min_ind - 1].max_ind += 1

            if trans_point.is_src_point():
                new_state[trans_point.max_ind + 1].min_ind = i + 1
            if trans_point.is_dest_point():
                new_state[trans_point.min_ind - 1].max_ind = i - 1

            new_state[i + 1:j + 1] = new_state[i:j]
            new_state[i] = trans_point
            successors.append(new_state)

        if len(successors) == 0:
            return None
        return min(successors, key=lambda state: self.get_state_cost(state))

    def _swap_indexes_successors_helper(self, state, i, j):
        first_point = state[i]
        last_point = state[j]
        if first_point.is_src_point() and j > first_point.max_ind:
            return -1
        if last_point.is_dest_point() and i < last_point.min_ind:
            return None

        new_state = copy_state(state)

        # update the index options of the relevant points:
        if first_point.is_src_point():
            new_state[first_point.max_ind + 1].min_ind = j + 1

        if first_point.is_dest_point():
            new_state[first_point.min_ind - 1].max_ind = j - 1

        if last_point.is_src_point():
            new_state[last_point.max_ind + 1].min_ind = i + 1
        if last_point.is_dest_point():
            new_state[last_point.min_ind - 1].max_ind = i - 1
        new_state[i], new_state[j] = new_state[j], new_state[i]

        return new_state

    def _create_successor(self, state, i, j, method):
        if method == INSERT_AND_SHIFT:
            return self._insert_and_shift_successors_helper(state, i, j)
        elif method == SWAP_ADJACENT:
            return self._swap_adjacent_successors_helper(state, i, i + 1)
        else:
            return self._swap_indexes_successors_helper(state, i, j)

    def get_random_successor(self, state, method=SWAP_INDEXES):
        n = len(state)
        rand_ind = np.random.choice(n, 2, replace=False)
        i, j = min(rand_ind), max(rand_ind)
        successor = self._create_successor(state, i, j, method=method)
        while successor is None:
            rand_ind = np.random.choice(n, 2, replace=False)
            i, j = min(rand_ind), max(rand_ind)
            successor = self._create_successor(state, i, j, method=method)
        return successor
        # return self._get_successors(state, random=True, method=method)

    def get_successors(self, state, method=SWAP_INDEXES, max_successors=None, random=False, generate_all=False):

        if random:
            return self.get_random_successor(state, method)

        successors = []
        min_cost = math.inf
        n = len(state)
        for i in range(n):
            for j in range(i + 1, n):
                if method == SWAP_ADJACENT:
                    first_ind, last_ind = j - 1, j
                else:
                    first_ind, last_ind = i, j
                if not generate_all:
                    cur_cost = self._get_cost_without_change(state, first_ind, last_ind, method)
                    if cur_cost == -1:  # method == SWAP_INDEXES *and* the swap is illegal
                        break
                    if cur_cost == math.inf:
                        continue
                    if cur_cost <= min_cost:
                        min_cost = cur_cost
                        new_state = self._create_successor(state, first_ind, last_ind, method)
                        if new_state is not None:
                            successors = [new_state]
                else:
                    new_state = self._create_successor(state, first_ind, last_ind, method=method)
                    if new_state == -1:  # method == SWAP_INDEXES *and* the swap is illegal
                        break
                    if new_state is not None:
                        successors.append(new_state)
            if method == SWAP_ADJACENT:
                break
        if not successors:
            return None
        return successors

    def get_state_cost(self, state, parts=False):
        if parts:
            state_cost = 0
            start_point = self.start_point
            for part in state:
                if len(part) == 0:
                    continue

                state_cost += start_point.distance(part[0])
                for i in range(len(part) - 1):
                    state_cost += part[i].distance(part[i + 1])
                start_point = part[-1]
            return state_cost
        else:
            state_cost = self.start_point.distance(state[0])
            for i in range(len(state) - 1):
                state_cost += state[i].distance(state[i + 1])
            return state_cost

    def get_random_states(self, k=1):
        perms = set()
        states = []
        for j in range(k):
            perm = tuple(np.random.permutation(len(self.points) // 2))
            while perm in perms:
                perm = tuple(np.random.permutation(len(self.points) // 2))
            perms.add(perm)
            random_state = []
            points = copy_state(self.points)
            for i in range(len(self.points) // 2):
                random_state += points[2 * perm[i]:2 * perm[i] + 2]
                random_state[2 * i].set_max_ind(2 * i)
                random_state[2 * i + 1].set_min_ind(2 * i + 1)
            states.append(random_state)
        return states if k > 1 else states[0]

    def _draw_sol(self, sol):
        x = [self.start_point.x]
        y = [self.start_point.y]
        for point in sol:
            x.append(point.x)
            y.append(point.y)
        for i in range(len(sol)):
            plt.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i],
                      shape='full', color='r', lw=1.6, length_includes_head=True,
                      zorder=0, head_length=.7, head_width=.3)

    def plot_progress(self, progress):
        fig = plt.figure(figsize=[14, 14])
        for sol in progress:
            x = [self.start_point.x]
            y = [self.start_point.y]
            for point in self.points:
                x.append(point.x)
                y.append(point.y)

            plt.scatter(x, y)
            plt.scatter(x[0], y[0], edgecolors='y', c='black', s=750, marker='X')
            annot = ["START"] + ["S", "D"] * (len(self.points) // 2)
            for i, point_type in enumerate(annot):
                plt.annotate(s=point_type, xy=(x[i], y[i]), xytext=(x[i] + 0.15, y[i] + 0.15))
            self._draw_sol(sol)
            xx = np.vstack([x[1::2], x[2::2]])
            yy = np.vstack([y[1::2], y[2::2]])
            plt.plot(xx, yy, ls="--", c="b", lw=0.4)
            plt.xticks(range(max_range + 1))
            plt.yticks(range(max_range + 1))
            plt.text(0.05, 0.95, "COST = " + str(self.get_state_cost(sol)), transform=plt.gca().transAxes)
            plt.draw()
            # plt.pause(1)
            plt.waitforbuttonpress(0)
            plt.clf()
        plt.close()


class UnitCapacityProblem(SearchProblem):

    def __init__(self, num_of_passengers=10, start_point=None, points=None):
        if start_point is not None:
            if isinstance(start_point, Point):
                self.start_point = start_point
            else:
                self.start_point = Point(start_point[0], start_point[1])
        else:
            self.start_point = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))

        if points is not None:
            self.points = import_from_points_array(points)
        else:
            self.points = []
            for i in range(num_of_passengers):
                p_s = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))
                p_d = Point(np.random.uniform(Box[0], Box[1]), np.random.uniform(Box[2], Box[3]))
                self.points += [p_s, p_d]
                p_d.add_src(p_s)
                p_s.add_dest(p_d)

    def get_start_state(self):
        return self.points

    def get_start_point(self):
        return self.start_point

    def _get_cost_without_change(self, state, i, j, method):
        n = len(state) // 2
        if method == INSERT_AND_SHIFT:
            parts = [state[:2 * i], state[2 * (i + 1):2 * (j + 1)], state[2 * i:2 * i + 2]]
            if j < n - 1:
                parts.append(state[2 * (j + 1):])
            return self.get_state_cost(parts, parts=True)

        elif method == SWAP_ADJACENT:
            parts = [state[:2 * i], state[2 * j:2 * j + 2], state[2 * i:2 * i + 2]]
            if j < n - 1:
                parts.append(state[2 * (j + 1):])
            return self.get_state_cost(parts, parts=True)
        else:
            parts = [state[:2 * i], state[2 * j:2 * j + 2], state[2 * (i + 1):2 * j], state[2 * i:2 * i + 2]]
            if j < n - 1:
                parts.append(state[2 * (j + 1):])
            return self.get_state_cost(parts, parts=True)

    def _swap_adjacent_successors_helper(self, state, i, j):
        new_state = state.copy()
        new_state[2 * i:2 * i + 2], new_state[2 * j:2 * j + 2] = new_state[2 * j:2 * j + 2], new_state[2 * i:2 * i + 2]
        return new_state

    def _insert_and_shift_successors_helper(self, state, i, j):
        new_state = state.copy()
        trans_point = new_state[2 * i:2 * i + 2]
        new_state[2 * i:2 * j] = new_state[2 * (i + 1):2 * (j + 1)]
        new_state[2 * j:2 * j + 2] = trans_point
        return new_state

    def _swap_indexes_successors_helper(self, state, i, j):
        new_state = state.copy()
        new_state[2 * i: 2 * i + 2], new_state[2 * j: 2 * j + 2] = state[2 * j: 2 * j + 2], state[2 * i: 2 * i + 2]
        return new_state

    def _create_successor(self, state, i, j, method):
        if method == INSERT_AND_SHIFT:
            return self._insert_and_shift_successors_helper(state, i, j)
        elif method == SWAP_ADJACENT:
            return self._swap_adjacent_successors_helper(state, i, i + 1)
        else:
            return self._swap_indexes_successors_helper(state, i, j)

    def get_random_successor(self, state, method=SWAP_INDEXES):
        n = len(state) // 2
        rand_ind = np.random.choice(n, 2, replace=False)
        i, j = min(rand_ind), max(rand_ind)
        successor = self._create_successor(state, i, j, method=method)
        while successor is None:
            rand_ind = np.random.choice(n, 2, replace=False)
            i, j = min(rand_ind), max(rand_ind)
            successor = self._create_successor(state, i, j, method=method)
        return successor

    def get_successors(self, state, method=SWAP_INDEXES, max_successors=None, random=False, generate_all=False):

        if random:
            return self.get_random_successor(state, method)

        successors = []
        min_cost = math.inf
        n = len(state) // 2
        for i in range(n):
            for j in range(i + 1, n):
                if method == SWAP_ADJACENT:
                    first_ind, last_ind = j - 1, j
                else:
                    first_ind, last_ind = i, j
                if not generate_all:
                    cur_cost = self._get_cost_without_change(state, first_ind, last_ind, method)
                    if cur_cost == -1:  # method == SWAP_INDEXES *and* the swap is illegal
                        break
                    if cur_cost == math.inf:
                        continue
                    if cur_cost <= min_cost:
                        min_cost = cur_cost
                        new_state = self._create_successor(state, first_ind, last_ind, method)
                        if new_state is not None:
                            successors = [new_state]
                else:
                    new_state = self._create_successor(state, first_ind, last_ind, method=method)
                    if new_state == -1:  # method == SWAP_INDEXES *and* the swap is illegal
                        break
                    if new_state is not None:
                        successors.append(new_state)
            if method == SWAP_ADJACENT:
                break

        return successors

    def get_state_cost(self, state, parts=False):
        if parts:
            state_cost = 0
            start_point = self.start_point
            for part in state:
                if len(part) == 0:
                    continue

                state_cost += start_point.distance(part[0])
                for i in range(len(part) - 1):
                    state_cost += part[i].distance(part[i + 1])
                start_point = part[-1]
            return state_cost
        else:
            state_cost = self.start_point.distance(state[0])
            for i in range(len(state) - 1):
                state_cost += state[i].distance(state[i + 1])
            return state_cost

    def get_random_states(self, k=1):
        perms = set()
        states = []
        for j in range(k):
            perm = tuple(np.random.permutation(len(self.points) // 2))
            while perm in perms:
                perm = tuple(np.random.permutation(len(self.points) // 2))
            perms.add(perm)
            random_state = []
            points = copy_state(self.points)
            for i in range(len(self.points) // 2):
                random_state += points[2 * perm[i]:2 * perm[i] + 2]
                # random_state[2 * i].set_max_ind(2 * i)
                # random_state[2 * i + 1].set_min_ind(2 * i + 1)
            states.append(random_state)
        return states if k > 1 else states[0]


def brute_force(problem):
    init_state = problem.get_start_state()
    fringe = Stack()
    fringe.push(tuple(init_state))
    closed = set()
    best_score = math.inf
    best_sol = None
    worst_cost = 0
    while not fringe.isEmpty():
        current_state = fringe.pop()
        cur_score = problem.get_state_cost(current_state)
        if cur_score < best_score:
            best_score = cur_score
            best_sol = current_state
        if cur_score > worst_cost:
            worst_cost = cur_score
        if current_state not in closed:  # no need to expand node that was already expanded before
            successors = problem.get_successors(list(current_state))

            for state in successors:
                fringe.push(tuple(state))
            closed.add(current_state)

    print("worst cost: ", worst_cost)
    return best_sol, best_score


# np.random.seed(3)


def is_valid_sol(sol, capacity=INF):
    if capacity == INF:
        for i in range(len(sol)):
            if sol[i].is_src_point():
                dest_ind = sol[i].max_ind + 1

                if sol[dest_ind] not in sol[i].dest_list:
                    print("src problem")
                    print("src: ", i, "dest: ", dest_ind)
                    return False
            else:
                src_ind = sol[i].min_ind - 1
                if sol[src_ind] not in sol[i].src_list:
                    print("dest problem")
                    print("src: ", src_ind, "dest: ", i)
                    return False
        return True
    else:
        for i in range(len(sol) // 2):
            if not sol[2 * i].is_src_point() or (sol[2 * i + 1] not in sol[2 * i].dest_list):
                return False
            if not sol[2 * i + 1].is_dest_point() or (sol[2 * i] not in sol[2 * i + 1].src_list):
                return False
        return True


def run(num_of_trials=1, file_name="results.txt", range_of_passengers=range(5, 7), capacity=INF, points=None,
        start_point=None, algos={}, write=False, brute=False, mode="sol", print_result=True):
    if mode == "sol":
        if num_of_trials != 1:
            print("in solution mode num_of_trial must be 1")
            return
        if len(algos) != 0 and (len(algos) != 1 or (
                 GENETIC not in algos and len(algos[list(algos.keys())[0]]) != 1)):
            print("In solution mode one algorithm and one method or set of params should be given")
            return
    costs = {}
    times = {}
    costs[GREEDY] = [0] * len(range_of_passengers)
    times[GREEDY] = [0] * len(range_of_passengers)
    for a in algos:
        methods = algos[a]
        for m in methods:
            new_m = m
            if isinstance(m, list):
                new_m = tuple(m)
            costs[(a, new_m)] = [0] * len(range_of_passengers)
            times[(a, new_m)] = [0] * len(range_of_passengers)
    if brute:
        costs[BRUTE] = [0] * len(range_of_passengers)
        times[BRUTE] = [0] * len(range_of_passengers)

    range_count = 0
    if points is not None and len(range_of_passengers) != 1:
        print("if points gives as an input, num of passengers must be constant")
        return

    for n in range_of_passengers:
        sol = []
        cost = 0
        for i in range(num_of_trials):
            if points is None:
                points, start_point = crateRandomProblem(n)
            # print("Trial:", i, "n:", n)
            tot_sols = []
            text = []
            if capacity == ONE:
                my_problem = UnitCapacityProblem(num_of_passengers=n, points=points, start_point=start_point)
            else:
                my_problem = InfCapacityProblem(num_of_passengers=n, points=points, start_point=start_point)

            if mode == 'sol':
                if len(algos) == 0:
                    greedy_sol = greedy_search(my_problem, capacity=capacity)
                    greedy_cost = my_problem.get_state_cost(greedy_sol)
                    # print(is_valid_sol(greedy_sol,capacity))
                    if print_result:
                        print("SOLUTION:")
                        print_state(greedy_sol)
                        print("COST:", greedy_cost)
                    return
            else:
                start_time = time.time()
                greedy_sol = greedy_search(my_problem, capacity=capacity)
                times[GREEDY][range_count] += time.time() - start_time
                greedy_cost = my_problem.get_state_cost(greedy_sol)
                costs[GREEDY][range_count] += greedy_cost
                tot_sols.append(greedy_sol)
                text += ["#TRIAL#", "Num Of Passengers: " + str(n),
                         "Input: " + print_state(my_problem.points, show=False)]
                text += ["greedy sol: " + str(greedy_cost)]

            if brute:
                start_time = time.time()
                best_sol, best_cost = brute_force(my_problem)
                costs[BRUTE][range_count] += best_cost
                times[BRUTE][range_count] += time.time() - start_time
                tot_sols.append(best_sol)
                text += ["Optimal solution:", print_state(best_sol, show=False), "Cost: " + str(best_cost)]

            if HILL_CLIMBING in algos:
                for method in algos[HILL_CLIMBING]:
                    start_time = time.time()
                    sols = hill_climbing_search(my_problem, successors_method=method)
                    times[(HILL_CLIMBING, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sols[-1])
                    costs[(HILL_CLIMBING, method)][range_count] += cost
                    tot_sols.append(sols[-1])
                    text += [HILL_CLIMBING + method,
                             "START" + str(my_problem.start_point) + " " + print_state(sols[-1], show=False),
                             "Cost: " + str(cost)]

            if SIMULATED_ANNEALING in algos:
                for method in algos[SIMULATED_ANNEALING]:
                    start_time = time.time()
                    sol = simulated_annealing_search(my_problem, successors_method=method)
                    times[(SIMULATED_ANNEALING, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(SIMULATED_ANNEALING, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [SIMULATED_ANNEALING + method,
                             str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if BEAM in algos:
                for method in algos[BEAM]:
                    start_time = time.time()
                    sols = local_beam_search(my_problem, successors_method=method, k=3)
                    times[(BEAM, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sols[-1])
                    costs[(BEAM, method)][range_count] += cost
                    tot_sols.append(sols[-1])
                    text += [BEAM + method, str(my_problem.start_point) + print_state(sols[-1], show=False),
                             "Cost: " + str(cost)]

            if STOCHASTIC_BEAM in algos:
                for method in algos[STOCHASTIC_BEAM]:
                    start_time = time.time()
                    sol = stochastic_local_beam_search(my_problem, successors_method=method, k=3)
                    times[(STOCHASTIC_BEAM, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(STOCHASTIC_BEAM, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [STOCHASTIC_BEAM + method, str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if TABU in algos:
                for method in algos[TABU]:
                    start_time = time.time()
                    sol = tabu_search(my_problem, successors_method=method, max_tabu_size=4)
                    times[(TABU, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(TABU, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [TABU + method, str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if LATE_ACCEPTANCE in algos:
                for method in algos[LATE_ACCEPTANCE]:
                    start_time = time.time()
                    sol = late_acceptance_hill_climbing(my_problem, successors_method=method)
                    times[(LATE_ACCEPTANCE, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(LATE_ACCEPTANCE, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [LATE_ACCEPTANCE + method, str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if RESTART_HILL in algos:
                for method in algos[RESTART_HILL]:
                    start_time = time.time()
                    sol = random_restart_hill_climbing(my_problem, successors_method=method, k=4)
                    times[(RESTART_HILL, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(RESTART_HILL, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [RESTART_HILL + method, str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if FIRST_CHOICE in algos:
                for method in algos[FIRST_CHOICE]:
                    start_time = time.time()
                    sol = first_choice_hill_climbing(my_problem, successors_method=method)
                    times[(FIRST_CHOICE, method)][range_count] += time.time() - start_time
                    cost = my_problem.get_state_cost(sol)
                    costs[(FIRST_CHOICE, method)][range_count] += cost
                    tot_sols.append(sol)
                    text += [FIRST_CHOICE + method, str(my_problem.start_point) + print_state(sol, show=False),
                             "Cost: " + str(cost)]

            if GENETIC in algos:
                if capacity == ONE:
                    for params in algos[GENETIC]:
                        start_time = time.time()
                        sol = GA(points, start_point, params[0], params[1], params[2], params[3], params[4])
                        times[(GENETIC, tuple(params))][range_count] += time.time() - start_time
                        sol = import_from_points_array(sol[1:, :])
                        cost = my_problem.get_state_cost(sol)
                        costs[(GENETIC, tuple(params))][range_count] += cost
                        tot_sols.append(sol)
                if capacity == INF:
                    for params in algos[GENETIC]:
                        start_time = time.time()
                        sol = inf.GA(points, start_point, params[0], params[1], params[2], params[3])
                        times[(GENETIC, tuple(params))][range_count] += time.time() - start_time
                        sol = import_from_points_array(sol[1:, :])
                        cost = my_problem.get_state_cost(sol)
                        costs[(GENETIC, tuple(params))][range_count] += cost
                        tot_sols.append(sol)

            # check validity:
            for sols in tot_sols:

                if not is_valid_sol(sols, capacity=capacity):
                    print("Solution is not valid, trial:", i)
                    print_state(sols)
                    print("Initial State:")
                    print_state(my_problem.points)
                    exit(1)

            if write:
                f = open(file_name, "a")
                for info in text:
                    f.write(info + '\n')
                f.write("- - - - - - - - - - - - - - - - -\n")
                f.close()
            points, start_point = None, None

            if mode == "sol":
                if print_result:
                    print("SOLUTION:")
                    print_state(sol)
                    print("COST:", cost)
                return sol, cost

        for key in costs:
            costs[key][range_count] /= num_of_trials
            times[key][range_count] /= num_of_trials

        range_count += 1
    return costs, times


# exp_list = {HILL_CLIMBING: [SWAP_ADJACENT, INSERT_AND_SHIFT], RESTART_HILL: [INSERT_AND_SHIFT],
#             TABU: [INSERT_AND_SHIFT], LATE_ACCEPTANCE: [INSERT_AND_SHIFT]}
# exp_list = {HILL_CLIMBING: [SWAP_ADJACENT, INSERT_AND_SHIFT], RESTART_HILL: [INSERT_AND_SHIFT],
#             TABU: [INSERT_AND_SHIFT], GENETIC: [(100, 0.8, 0.7, 50, 0), (100, 0.8, 0.7, 50, 1)]}
# my_range = range(5, 10)
# costs, times = run(num_of_trials=10, range_of_passengers=my_range, capacity=ONE, brute=True, algos=exp_list)

def plot_results(results, visualization="bar"):
    costs, times, my_range, num_of_trials, exp_list = results
    if visualization == "line":

        plt.figure(figsize=[6, 8])
        for res in costs:
            plt.plot(my_range, costs[res], label=res)
            plt.legend(loc="upper left", fontsize=10)
        plt.xlabel('NUMBER OF PASSENGERS')
        plt.xticks(my_range, fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("COST", fontsize=10)
        plt.savefig("costs.png")

        plt.figure(figsize=[6, 8])
        for res in times:
            plt.plot(my_range, times[res], label=res)
            plt.legend(loc="upper left", fontsize=10)
        plt.xlabel('NUMBER OF PASSENGERS')
        plt.xticks(my_range, fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("RUNTIME [SEC]", fontsize=10)
        plt.savefig("times.png")
    else:
        barWidth = 0.22
        fig = plt.subplots(figsize=(35, 12))
        i = 0
        for res in costs:
            plt.bar(np.arange(len(my_range)) + barWidth * i, costs[res], width=barWidth,
                    edgecolor='grey', label=res)

            for j in range(len(my_range)):
                plt.annotate(str(np.format_float_positional(costs[res][j], precision=2)),
                             xy=(j + barWidth * i, costs[res][j]), ha='center', va='bottom')
            i += 1
        plt.xlabel('NUMBER OF PASSENGERS', fontweight='bold', fontsize=15)
        plt.ylabel('COST', fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(my_range))], my_range, fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig("costs.png")

        fig = plt.subplots(figsize=(35, 12))
        i = 0
        for res in costs:
            plt.bar(np.arange(len(my_range)) + barWidth * i, times[res], width=barWidth,
                    edgecolor='grey', label=res)

            for j in range(len(my_range)):
                plt.annotate(str(np.format_float_positional(times[res][j], precision=2)),
                             xy=(j + barWidth * i, times[res][j]), ha='center', va='bottom')
            i += 1
        plt.xlabel('NUMBER OF PASSENGERS', fontweight='bold', fontsize=15)
        plt.ylabel('RUNTIME [SEC]', fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(my_range))], my_range, fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig("times.png")

import numpy as np
from math import sin, cos, sqrt, atan2, radians
Box = (35.1694, 35.2487,
       31.7493, 31.8062)


class Point:

    def __init__(self, x=0, y=0):

        self.x = x
        self.y = y
        self.dest_list = set()
        self.src_list = set()
        self.max_ind = 0
        self.min_ind = 0

    def add_dest(self, dest):
        self.dest_list.add(dest)

    def add_src(self, src):
        self.src_list.add(src)

    def set_max_ind(self, ind):
        self.max_ind = ind

    def set_min_ind(self, ind):
        self.min_ind = ind

    def is_src_point(self):
        return len(self.dest_list) != 0

    def is_dest_point(self):
        return len(self.src_list) != 0

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str((self.x, self.y))

    def __deepcopy__(self):
        # cls = self.__class__
        # result = cls.__new__(cls)
        p_cpy = Point(self.x, self.y)
        p_cpy.min_ind = self.min_ind
        p_cpy.max_ind = self.max_ind
        if self.is_src_point():
            dest_point = list(self.dest_list)[0]
            new_dest = Point(dest_point.x, dest_point.y)
            new_dest.min_ind = dest_point.min_ind
            new_dest.max_ind = dest_point.max_ind
            new_dest.src_list.add(self)
            p_cpy.dest_list.add(new_dest)
        if self.is_dest_point():
            src_point = list(self.src_list)[0]
            new_src = Point(src_point.x, src_point.y)
            new_src.min_ind = src_point.min_ind
            new_src.max_ind = src_point.max_ind
            new_src.dest_list.add(self)
            p_cpy.src_list.add(new_src)

        # p_cpy.src_list = self.src_list
        # p_cpy.des_list = self.dest_list
        return p_cpy

        # for k, v in self.__dict__.items():
        #     setattr(result, k, deepcopy(v))
        # return result

    def distance(self, other):
        R = 6373.0
        if self.x == np.inf or other.x == np.inf:
            return np.inf
        lat1 = np.radians(self.x)
        lon1 = np.radians(self.y)
        lat2 = np.radians(other.x)
        lon2 = np.radians(other.y)
        """
        returns the geometric distance we could add the google maps min road distance in the future
        """
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        return distance
        # return np.sqrt(abs(self.x - other.x)**2 + abs(self.y - other.y)**2)


class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


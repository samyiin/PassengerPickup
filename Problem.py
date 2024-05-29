import matplotlib.pyplot as plt
import numpy as np
import random
import math
from math import sin, cos, sqrt, atan2, radians

# number of sits in our bus, vary from 1 to inf, very important value
NOS = 1
# our jerusalem in lat, long
Box = (35.1694, 35.2487,
       31.7493, 31.8062)


def d(p1, p2):
    R = 6373.0
    if p1[0] == np.inf or p2[0] == np.inf:
        return np.inf
    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])
    """
    returns the geometric distance we could add the google maps min road distance in the future
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def crateRandomProblem(n: int):
    """
    :param n:
    the number of people requests
    :return:
    should return numpy arrey with n rows 4 columns (lat1,long1,lat2,long2), from the locations of jerusalem
    """
    a = np.zeros((n, 4))
    a = np.apply_along_axis(lambda x: (
    random.uniform(Box[0], Box[1]), random.uniform(Box[2], Box[3]), random.uniform(Box[0], Box[1]),
    random.uniform(Box[2], Box[3])), axis=1, arr=a)
    return a, (35.203958315434434, 31.788860911945207)


def DrawProblem(points, bus=(35.203958315434434, 31.788860911945207)):
    """
    prints a map with all the dots on it and the dot of our bus
    https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db
    """
    im = plt.imread('jerusalem.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(points[:, 0], points[:, 1], zorder=1, c='b', s=10)
    ax.scatter(points[:, 2], points[:, 3], zorder=1, c='r', s=10)
    ax.scatter(bus[0], bus[1], zorder=1, marker="s", c='g', s=30)
    ax.set_title('Plotting Spatial Data on jerusalem Map')
    ax.set_xlim(Box[0], Box[1])
    ax.set_ylim(Box[2], Box[3])
    ax.imshow(im, zorder=0, extent=Box, aspect='equal')
    plt.show()


def randomSolution(points, bus=(35.203958315434434, 31.788860911945207)):
    n = points.shape[0]
    sel = np.zeros((2 * n + 1, 2))
    sel[0] = bus
    rper = np.random.permutation(n)
    sel[range(1, 2 * n + 1, 2)] = points[rper, 0:2]
    sel[range(2, 2 * n + 1, 2)] = points[rper, 2:4]

    return sel


def score(points):
    shiftp = np.zeros((len(points), 2))
    shiftp[0:-1] = points[1:]
    shiftp[-1] = points[-1]
    dpoints = np.hstack([points, shiftp])
    return sum(np.apply_along_axis(lambda p: d(p[0:2], p[2:4]), axis=1, arr=dpoints))


def DrawSolution(points):
    """
    prints a map with all the dots on it and the dot of our bus
    https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db
    """
    im = plt.imread('jerusalem.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(0, len(points) + 1):
        plt.plot(points[i:i + 2, 0], points[i:i + 2, 1], 'ro-')
    ax.set_title('Plotting route on jerusalem Map')
    ax.set_xlim(Box[0], Box[1])
    ax.set_ylim(Box[2], Box[3])
    ax.imshow(im, zorder=0, extent=Box, aspect='equal')
    plt.show()

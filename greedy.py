import Problem
import numpy as np
def greedySeloution(points,bus):
    n=points.shape[0]
    sel=np.zeros((2*n+1,2))
    sel[0]=bus

    f=lambda x,y: Problem.d(x[0:2],y)
    for i in range(n):
        arg=np.argmin(np.fromiter((f(xi,sel[2*i]) for xi in points), points.dtype))
        sel[2*i+1]=points[arg,0:2]
        sel[2 * i + 2] = points[arg, 2:4]
        points[arg]=[np.inf]*4
    return sel
import numpy as np

from Problem import *
import itertools


def bruteFroce(points, bus):
    n=points.shape[0]
    sel=np.zeros((2*n+1,2))
    sel[0]=bus

    perms=itertools.permutations(range(n))
    shiftp=np.zeros((len(points),2))
    curmin=np.inf
    curper=0
    for per in perms:
        cur=points[per,:]
        shiftp[0:-1]=cur[1:,0:2]
        shiftp[-1]=cur[-1,2:4]
        dpoints=np.hstack([cur,shiftp])
        m=sum(np.apply_along_axis(lambda p: d(p[0:2], p[2:4])+d(p[4:6], p[2:4]), axis=1, arr=dpoints))+d(bus,cur[0,0:2])
        if m<curmin:
            curper=per
            curmin=m
    sel[range(1,2*n+1,2)]=points[curper,0:2]
    sel[range(2, 2 * n +1, 2)] = points[curper, 2:4]
    return sel
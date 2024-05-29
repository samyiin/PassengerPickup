import random
from operator import itemgetter
import CrossoverFunction
import matplotlib.pyplot as plt

from Problem import *
def Crossover(path1,path2,r,type):
    path1, path2 = CrossoverFunction.crossover(path1,path2,r,type)
    return [path1,path2]

def mutate(path,r):
    # return mutate in r probality or only the path
    c=random.choices([0,1],[1-r,r])

    if c:
        a=random.randint(0,path.shape[0]-1)
        b=random.randint(0,path.shape[0]-1)
        path[a,:],path[b,:]=path[b,:],path[a,:].copy()
    return path

#get a path and return the distance of it
def evlotion(path,b):
    shiftp = np.zeros((len(path), 2))
    shiftp[0:-1]=path[1:,0:2]
    shiftp[-1]=path[-1,2:4]
    dpoints=np.hstack([path,shiftp])
    m=sum(np.apply_along_axis(lambda p: d(p[0:2], p[2:4])+d(p[4:6], p[2:4]), axis=1, arr=dpoints))+d(b,path[0,0:2])
    return m

#gets the scors of each path and returning what will be his precntge in the genretion
def precentge(l):
    s=sum(l)/(2*len(l))
    s=[(math.exp((-(sc-min(l))/2))) for sc in l]
    return [float(i) / sum(s) for i in s]

#gets poplation size, Crossover rate, mutate rate, Number of genrations
def GA(p,b,PS,Cr,Mr,GN,type=0):
    n=p.shape[0]
    Poplation=[np.random.permutation(p) for _ in range(PS)]
    #for debug
    Bestdistace=[]
    bestpath=[]
    for cur_gen_num in range(GN):
        scors=[evlotion(p,b) for p in Poplation]
        index, element = min(enumerate(scors), key=itemgetter(1))
        bestpath.append(Poplation[index])
        Bestdistace.append(element)

        pscors=precentge(scors)
        #croos over
        perents=[random.choices(Poplation,pscors,k=2) for _ in range(int(PS/2))]
        Poplation=[]
        for pair in perents:
            Poplation.extend(Crossover(pair[0],pair[1],Cr,type))

        #mutate
        for path in Poplation:
            path=mutate(path,Mr)

        if cur_gen_num == GN // 4:
            print("25%")
        if cur_gen_num == GN // 2:
            print("50%")
        if cur_gen_num == 3 * GN // 4:
            print("75%")

    scors = [evlotion(p,b) for p in Poplation]
    index, element = min(enumerate(scors), key=itemgetter(1))
    bestpath.append(Poplation[index])
    Bestdistace.append(element)
    # plt.plot(range(len(Bestdistace)),Bestdistace)
    # plt.show()
    index, element = min(enumerate(Bestdistace), key=itemgetter(1))
    theBest=bestpath[index]

    sel=np.zeros((2*n+1,2))
    sel[0]=b
    sel[range(1, 2 * n + 1, 2)] = theBest[:, 0:2]
    sel[range(2, 2 * n +1, 2)] = theBest[:, 2:4]

    return sel

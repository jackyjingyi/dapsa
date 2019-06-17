import numpy as np


PTS = 0.2


def approxiamtion(xc,x0,width=3,pts = PTS):
    return abs(xc-x0) < width *(1+pts)


def inside(obj1,obj2,width =3, pts =0.2):
    matrix = np.array(obj1.bbox)>np.array(obj2.bbox)
    distinct = np.array(obj1.bbox)>(np.array(obj2.bbox)+
            np.array((width*(1+pts),width*(1+pts),0-width*(1+pts),0-width*(1+pts))))
    if not any(matrix[0:2]) and all(matrix[2:]):
        return True
    else:
        if not any(distinct[0:2]) and all(distinct[2:]):
            return True
        else:
            return False

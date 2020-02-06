# this function is for initializing parameters with zeros
import numpy as np


def initialize_with_zeros(dim):
    #from NN structure defined in main function, w with shape(dim,1) b with shape(1,1)
    w = np.zeros((dim,1))
    b = 0
    return w,b
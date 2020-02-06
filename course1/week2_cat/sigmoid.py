# helper function to calculate sigmoid of input z
import numpy as np

def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A
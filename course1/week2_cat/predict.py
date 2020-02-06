# function to predict ouput Y_hat with optimized parameters
import numpy as np
from sigmoid import sigmoid


def predict(W,b,X):
    m = X.shape[1]
    Y_hat = np.zeros((1,m))
    Y_output = sigmoid(np.dot(W.T,X)+b)
    for i in range (m):
        if Y_output[0,i] >= 0.5:
            Y_hat[0,i] = 1
        else:
            Y_hat[0,i] = 0
    return Y_hat



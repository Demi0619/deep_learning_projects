import numpy as np
from sigmoid import sigmoid

def propagate(W,b,X,Y):
    """
    Implement forward (compute cost) and backward (compute dw,db) propagation
    :param W: weight parameters with shape(n_h*n_w*n_c,1)
    :param b: scalar
    :param X: input X with shape(n_h*n_w*n_c,m)
    :param Y: output Y with shape(1,m)
    :return: cost,dW,db
    """
    m = X.shape[1]
    #forward propagation
    Z = np.dot(W.T,X)+b  #linear (1,m)
    A = sigmoid(Z) #activation (1,m)
    cost = -(1/m)*sum(sum(Y*np.log(A)+(1-Y)*np.log(1-A)))
    #backward propagation
    dZ = A-Y #(1,m)
    dW = (1/m)*np.dot(X,dZ.T) #(n_h*n_w*n_c,1)
    db = (1/m)*sum(sum(dZ)) #(1,1)

    return cost,dW,db




'''
this part includes all seperates functions needed in deeper NN model
'''
import numpy as np
from dnn_utils_v2 import relu,sigmoid,relu_backward
# 1.parameter initialization


def initialize_para(layer_dimension):
    #layer_dimension is a list with number of units in each layer. include input layer
    parameters = {}
    L = len(layer_dimension)
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dimension[l],layer_dimension[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dimension[l],1))
    return parameters

# 2. forward propogation
# 2.1 linear+relu


def linear_relu(A_previous, W, b):
    #A_previous is the outoupt of previous layer, W,b is the parameter of current layer
    Z = np.dot(W,A_previous)+b
    A,_ = relu(Z)
    return A,Z
# 2.2 linear+sigmoid


def linear_sig(A_previous, W, b):
    Z = np.dot(W,A_previous)+b
    A,_ = sigmoid(Z)
    return A,Z
# 2.3 combine to forward propagation


def forward_propagation(X,parameters):
    L = int(len(parameters)/2) # L is the number of layers
    A_previous = X
    cache ={'A0':X,}
    for l in range(L-1):
        W,b = parameters['W'+str(l+1)],parameters['b'+str(l+1)]
        A,Z = linear_relu(A_previous, W, b)
        A_previous = A
        cache['A' + str(l+1)] = A_previous
        cache['Z'+str(l+1)] = Z
    A,cache['Z'+str(L)] = linear_sig(A_previous, parameters['W'+str(L)], parameters['b'+str(L)])
    return A, cache
# 3. compute cost


def compute_cost(A,Y):
    m = Y.shape[1]
    cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A),axis=1,keepdims=True)
    return cost
# 4. back propogation
# 4.1 back propagation for a single linear+relu unit


def back_relu(dA,Z,A,W):
    m = dA.shape[1]
    dZ = relu_backward(dA,Z)
    dw = (1/m)*np.dot(dZ,A.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA = np.dot(W.T,dZ)
    return dw,db,dA


def back_propogation(A,Y,cache,parameters):
    m = Y.shape[1]
    grad = {}
    dZ = A-Y
    L = int(len(parameters)/2)
    grad['dW' + str(L)] = (1/m)*np.dot(dZ,cache['A'+str(L-1)].T)
    grad['db' + str(L)] = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA = np.dot(parameters['W'+str(L)].T,dZ)
    for l in reversed(range(L-1)):
        dw,db,dA = back_relu(dA,cache['Z'+str(l+1)],cache['A'+str(l)],parameters['W'+str(l+1)])
        grad['dW'+str(l+1)] = dw
        grad['db'+str(l+1)] = db
    return grad
# 5. update parameters


def update_para(parameters, grad,learning_rate=0.9):
    L = int(len(parameters)/2)
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grad['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grad['db'+str(l+1)]
    return parameters

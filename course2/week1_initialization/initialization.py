import numpy as np
import matplotlib.pyplot as plt
from init_utils import load_dataset,forward_propagation,relu,sigmoid,backward_propagation,update_parameters,compute_loss, predict, plot_decision_boundary, predict_dec

# 1. load and check data
train_X, train_Y, test_X, test_Y = load_dataset()
#print(f'train_x shape: {train_X.shape}, train_y shape:{train_Y.shape}')
m_train = train_X.shape[1]
m_test = test_X.shape[1]

# 2.build NN model with different initialization
# 2.1 build different initialization functions


def zero_initialization(layer_dimension):
    parameter = {}
    for l in range(1,len(layer_dimension)):
        parameter['W'+str(l)] = np.zeros((layer_dimension[l],layer_dimension[l-1]))
        parameter['b'+str(l)] = np.zeros((layer_dimension[l],1))
    return parameter


def random_initialization(layer_dimension):
    parameters = {}
    for l in range(1,len(layer_dimension)):
        parameters['W'+str(l)] = np.random.randn(layer_dimension[l],layer_dimension[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dimension[l],1))
    return parameters


def he_initialization(layer_dimension):
    parameters = {}
    for l in range(1,len(layer_dimension)):
        parameters['W'+str(l)] = np.random.randn(layer_dimension[l],layer_dimension[l-1])*np.sqrt(2/(layer_dimension[l-1]))
        parameters['b'+str(l)] = np.zeros((layer_dimension[l],1))
    return parameters

# 2.2 build model


def ini_model(X, Y, layer_dimension, initialize_method, learning_rate, num_iteration=15000):
    if initialize_method == 'zero':
        parameters = zero_initialization(layer_dimension)
    if initialize_method =='random':
        parameters =random_initialization(layer_dimension)
    if initialize_method == 'he':
        parameters = he_initialization(layer_dimension)
    for i in range(0,num_iteration):
        A3,cache = forward_propagation(X,parameters)
        grads = backward_propagation(X,Y,cache)
        parameters = update_parameters(parameters,grads,learning_rate)
        cost = compute_loss(A3,Y)
        if i % 1000 == 0:
            print(f'cost after iteration {i}:{cost}')
    return parameters


layer_dimension = [train_X.shape[0],10,5,1]
# 3. measure different initialize methods(cost,train/test accuracy,decision boundry) with train/test dataset
#3.1 zero_initialization
print('zero_initialization')
parameters = ini_model(train_X,train_Y,layer_dimension,'zero',0.01)
print('train accuracy:')
predict(train_X, train_Y, parameters)
print('test accuracy:')
predict(test_X, test_Y, parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)
# 3.2 random initialization
print('random initialization')
parameters = ini_model(train_X,train_Y,layer_dimension,'random',0.01)
print('train accuracy:')
predict(train_X,train_Y,parameters)
print('test accuracy:')
predict(test_X,test_Y,parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)
# 3.3 he initialization
print('he initialization')
parameters = ini_model(train_X,train_Y,layer_dimension,'he',0.01)
print('train accuracy:')
predict(train_X,train_Y,parameters)
print('test accuracy:')
predict(test_X,test_Y,parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)

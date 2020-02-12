from reg_utils import load_2D_dataset,compute_cost,relu,sigmoid,initialize_parameters,forward_propagation,backward_propagation,update_parameters,predict,plot_decision_boundary,predict_dec
import numpy as np
# 1. load and check dataset
train_X, train_Y, test_X, test_Y = load_2D_dataset()
print(f'train_x:{train_X.shape}\n train_y:{train_Y.shape}\n test_x:{test_X.shape}')
m_train = train_X.shape[1]
m_test = test_X.shape[1]
# 2. build model with different regularization
# 2.1 different regularization functions


def compute_l2_cost(AL,Y,parameters,lamda):
    '''compute the l2 norm part for l2 cost
    '''
    L = int(len(parameters)/2)
    m = Y.shape[1]
    norm = 0
    for l in range (1,L+1):
        norm += np.sum(np.square(parameters['W'+str(l)]))
    l2_cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))+(lamda/(2*m))*norm
    return l2_cost


def l2_back_propagation(X,Y,AL,cache,lamda):
    '''compute gradients with l2 norm
    dW[l] = normal_gradients + (lambda/m)*W[l]'''
    L = int(len(cache)/4)
    m = Y.shape[1]
    l2_gradients = {}
    l2_gradients['dZ'+str(L)] = AL-Y
    l2_gradients['dW'+str(L)] = (1/m)*np.dot(l2_gradients['dZ'+str(L)],cache['A'+str(L-1)].T)+(lamda/m)*cache['W'+str(L)]
    l2_gradients['db'+str(L)] = (1/m)*np.sum(l2_gradients['dZ'+str(L)],axis=1,keepdims=True)
    cache['A'+str(0)] = X
    for l in reversed(range(1,L)):
        l2_gradients['dA'+str(l)] = np.dot(cache['W'+str(l+1)].T,l2_gradients['dZ'+str(l+1)])
        l2_gradients['dZ'+str(l)] = np.multiply(l2_gradients['dA'+str(l)], np.int64(cache['A'+str(l)] > 0))
        l2_gradients['dW'+str(l)] = (1/m)*np.dot(l2_gradients['dZ'+str(l)],cache['A'+str(l-1)].T)+(lamda/m)*cache['W'+str(l)]
        l2_gradients['db' + str(l)] = (1/m)* np.sum(l2_gradients['dZ' + str(l)], axis=1, keepdims=True)
    return l2_gradients


def drop_forward(X,parameters,keep_prob):
    ''' forward process with dropout
        A[l] = A[l]*d[l] (d[l] is array with value (0/1) same size as A[l] '''
    L = int(len(parameters)/2)
    cache ={}
    cache['A'+str(0)] = X
    for l in range(1,L):
        cache['Z'+str(l)] = np.dot(parameters['W'+str(l)],cache['A'+str(l-1)])+parameters['b'+str(l)]
        cache['A'+str(l)] = relu(cache['Z'+str(l)])
        cache['d'+str(l)] = (np.random.rand(cache['A'+str(l)].shape[0],cache['A'+str(l)].shape[1]) < keep_prob).astype(int)
        cache['A'+str(l)] = cache['A'+str(l)] * cache['d'+str(l)]
        cache['A'+str(l)] = cache['A'+str(l)]/ keep_prob
        cache['W'+str(l)] = parameters['W'+str(l)]
        cache['b'+str(l)] = parameters['b'+str(l)]
    cache['Z'+str(L)] = np.dot(parameters['W'+str(L)],cache['A'+str(L-1)])+parameters['b'+str(L)]
    cache['A' + str(L)] = sigmoid(cache['Z' + str(L)])
    cache['W'+str(L)] = parameters['W'+str(L)]
    cache['b'+str(L)] = parameters['b'+str(L)]
    return cache['A'+str(L)],cache


def back_dropout(Y,AL,cache,keep_prob):
    ''' compute gradients with dropout
        dA[l] = dA[l]*d[l]'''
    L = int(len(cache)/5)
    m = Y.shape[1]
    gradients_drop = {}
    gradients_drop['dZ'+str(L)] = AL-Y
    gradients_drop['dW'+str(L)] = (1/m)*np.dot(gradients_drop['dZ'+str(L)],cache['A'+str(L-1)].T)
    gradients_drop['db'+str(L)] = (1/m)*np.sum(gradients_drop['dZ'+str(L)],axis=1, keepdims=True)
    for l in reversed(range(1,L)):
        gradients_drop['dA'+str(l)] = np.dot(cache['W'+str(l+1)].T,gradients_drop['dZ'+str(l+1)])
        gradients_drop['dA'+str(l)] = gradients_drop['dA'+str(l)] * cache['d'+str(l)]
        gradients_drop['dA'+str(l)] = gradients_drop['dA'+str(l)] / keep_prob
        gradients_drop['dZ'+str(l)] = np.multiply(gradients_drop['dA'+str(l)], np.int64(cache['A'+str(l)] > 0))
        gradients_drop['dW'+str(l)] = (1/m)*np.dot(gradients_drop['dZ'+str(l)],cache['A'+str(l-1)].T)
        gradients_drop['db'+str(l)] = (1/m)*np.sum(gradients_drop['dZ'+str(l)],axis=1,keepdims=True)
    return gradients_drop


def model(X,Y,layer_dims,keep_prob,lamda,learning_rate,num_iteration,norm=False):
    parameters = initialize_parameters(layer_dims)
    if norm == False and keep_prob == 1:
        for i in range(num_iteration):
            AL,cache = forward_propagation(X,parameters)
            grads = backward_propagation(X, Y, cache)
            parameters = update_parameters(parameters, grads, learning_rate)
            if i%1000 == 0:
                cost = compute_cost(AL,Y)
                print(f'cost after {i} iteration:{cost}')
        return parameters
    if norm == False and keep_prob != 1:
        for i in range(num_iteration):
            AL,cache = drop_forward(X,parameters,keep_prob)
            grads = back_dropout(Y,AL,cache,keep_prob)
            parameters = update_parameters(parameters,grads,learning_rate)
            if i%1000 ==0:
                cost = compute_cost(AL,Y)
                print(f'cost after{i} iterations:{cost}')
        return parameters
    if norm == True and keep_prob ==1:
        for i in range(num_iteration):
            AL,cache = forward_propagation(X,parameters)
            grads = l2_back_propagation(X,Y,AL,cache,lamda)
            parameters = update_parameters(parameters,grads,learning_rate)
            if i%1000 ==0:
                cost = compute_l2_cost(AL,Y,parameters,lamda)
                print(f'cost after {i} iteration:{cost}')
        return parameters

layer_dims = [train_X.shape[0],20,3,1]
#3.Evaluate different regularization methods
print('no regularization')
parameters = model(train_X,train_Y,layer_dims,1,0,0.3,30000,norm=False)
print('train set:')
predict(train_X,train_Y,parameters)
print('test set:')
predict(test_X,test_Y,parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)

print('with L2 norm regularization')
parameters = model(train_X,train_Y,layer_dims,1,0.1,0.3,30000,norm=True)
print('train set:')
predict(train_X,train_Y,parameters)
print('test set:')
predict(test_X,test_Y,parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)

print('with drop-out regularization')
parameters = model(train_X,train_Y,layer_dims,0.86,0,0.3,30000,norm=False)
print('train set:')
predict(train_X,train_Y,parameters)
print('test set:')
predict(test_X,test_Y,parameters)
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)













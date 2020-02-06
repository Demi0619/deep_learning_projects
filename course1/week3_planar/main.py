import numpy as np
from planar_utils import load_planar_dataset, plot_decision_boundary,sigmoid
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
np.random.seed(1)
# 1. pre-process the data
# 1.1 load and visualize the data
X,Y=load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c=Y[0,:],s=40,cmap=plt.cm.Spectral)
plt.show()
# 1.2 evaluate the data
m = X.shape[1]
print(f'X of shape {X.shape}, Y of shape{Y.shape}, have {m} examples in dataset')
# 2. fit and evaluate with logistic regression, a proof of why deeper NN is needed
# 2.1 fit the data with simple logistic regression model
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)
# 2.2 plot the decision bondary compute by simple regression model
plot_decision_boundary(lambda x: clf.predict(x),X,Y)
# 3. Define the NN structure


def layer_size(X,Y):
    n_x = X.shape[0]   #n_X refers to the number of input layer units
    n_y = Y.shape[0]   # n_Y refers to number of output layer units
    n_h = 4            # n_h refers to number of hidden layer units
    return (n_x,n_y,n_h)
# 4. forward and backward propagation
# 4.1 parameter initialization


def initialize_parameters(n_x,n_y,n_h):
    W1 = np.random.randn(n_h,n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)
    b2 = np.zeros((n_y,1))
    parameters = {'W1' : W1,
                  'W2' : W2,
                  'b1' : b1,
                  'b2' : b2}
    return parameters
# 4.2 forward propagation


def forward_propagation(X,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    Z1 = np.dot(W1,X)+b1 # (n_h,m)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2 #(n_y,m)
    A2 = sigmoid(Z2)
    cache={'Z1' : Z1,
           'A1' : A1,
           'Z2' : Z2,
           'A2' : A2}
    return cache, A2
# 4.3 compute cost


def compute_cost(A2,Y):
    m = Y.shape[1]
    cost = -(1/m)*sum(sum(Y*np.log(A2)+(1-Y)*np.log(1-A2)))
    return cost
# 4.4 backpropagtion


def back_propagation(X,Y,parameters,cache):
    W2 = parameters['W2']
    A2 = cache['A2']
    A1 = cache['A1']
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    gradients={'dW2':dW2,
               'db2':db2,
               'dW1':dW1,
               'db1':db1}
    return gradients
# 5. optimize (update parameters in epoches)


def optimize(gradients, parameters,learning_rate=0.09):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    return parameters
# 7. use trained parameter to predict


def predict(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    _,y_hat = forward_propagation(X,parameters)
    prediction = y_hat>0.5
    return prediction
# 6. combine to build a NN model


def nn_model(X,Y,epoches,learning_rate):
    n_x,n_y,n_h = layer_size(X, Y)
    parameters = initialize_parameters(n_x, n_y, n_h)
    for i in range (epoches):
        cache,A2= forward_propagation(X,  parameters)
        gradients = back_propagation(X, Y, parameters, cache)
        parameters = optimize(gradients, parameters, learning_rate=0.09)
    cost = compute_cost(A2, Y)
    print(cost,parameters)
    plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)


nn_model(X,Y,10000,learning_rate=1.2)

# 7.2 plot decision boundary with nn model







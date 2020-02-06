#function to update parameters in num_iterations
from propagate import propagate


def optimize(W,b,X,Y,num_iteration,learning_rate=0.009):
    for i in range (num_iteration):
        _,dW,db = propagate(W,b,X,Y)
        W = W-learning_rate*dW
        b = b-learning_rate*db
    return W,b


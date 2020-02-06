# model function return model structure and train/test accuracy

from initial_para import initialize_with_zeros
from optimize import optimize
from predict import predict
import numpy as np


def model(train_X, train_Y, test_X, test_Y, num_iteration, learning_rate=0.5):
    m = train_X.shape[1]
    dim = train_X.shape[0]
    W_ini, b_ini = initialize_with_zeros(dim)
    W, b = optimize(W_ini, b_ini, train_X, train_Y, num_iteration, learning_rate)
    train_Y_hat = predict(W, b, train_X)
    test_Y_hat = predict(W, b, test_X)
    train_accuracy = (1 - np.mean(abs(train_Y - train_Y_hat))) * 100
    test_accuracy = (1 - np.mean(abs(test_Y - test_Y_hat))) * 100
    return train_accuracy,test_accuracy

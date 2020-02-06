from load_data import load_data
from step import initialize_para, forward_propagation, back_propogation,update_para,compute_cost
import matplotlib.pyplot as plt

# 1. preprocessing data
# 1.1 load data
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
# 1.2 check data
#print(f'train_set_x with shape {train_set_x_orig.shape}, train_set_y with shape{train_set_y_orig.shape},test_set_x with shape{test_set_x_orig.shape}')
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
# 1.3 view data
index = 20
plt.imshow(train_set_x_orig[index,:,:,:])
plt.show()
print(f"y={train_set_y_orig[0,index]}, this is an {classes[train_set_y_orig[0,index]].decode('utf-8')} image")
# 1.4 flatten and standardize the data
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_X = train_set_x/255
test_X = test_set_x/255

# 2. build NN model with L layers


def nn_model(X, Y, layer_dimension,num_iterations,learning_rate):
    parameters = initialize_para(layer_dimension)
    for i in range(num_iterations):
        A,cache = forward_propagation(X,parameters)
        grad = back_propogation(A,Y,cache,parameters)
        parameters = update_para(parameters, grad,learning_rate=0.9)
    cost = compute_cost(A,Y)
    print(cost, parameters)


layer_dimension = [12288,20,7,5,1]
nn_model(train_X,train_set_y_orig,layer_dimension,300,learning_rate=0.0075)
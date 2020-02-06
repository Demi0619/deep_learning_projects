# main logic for data processing and train/evaluate model

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
from model import model

# 1. load and explore data
# 1.1 load dataset (load_dataset is a method defined to read train/test data from h5 file)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()
# 1.2 show an example image in dataset
index = 10
plt.imshow(train_set_x_orig[index])
print(f'y={train_set_y_orig[:,index]},this is {classes[np.squeeze(train_set_y_orig[:,index])].decode("utf-8")}')

# 1.3 explore dataset
m_train = train_set_x_orig.shape[0]
n_height = test_set_x_orig.shape[1]
n_width = test_set_x_orig.shape[2]
n_channel = test_set_x_orig.shape[3]
m_test = test_set_x_orig.shape[0]

print(m_train,m_test,n_height,n_channel)

# 1.4 flatten the traning/test input X to (n_h*n_w*n_c,m)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# 1.5 standardize the dataset by divide each channel by 255
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# 2. define NN structure (1 layer-output layer with 1 output unit)
# 3. call model with train/test dataset
train_accuracy, test_accuracy =  model(train_set_x,train_set_y_orig,test_set_x,test_set_y_orig,2000,0.5)
print(f'train accuracy ={train_accuracy}% /n test accuracy={test_accuracy}%')







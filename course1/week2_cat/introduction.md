This project is to identify an image as cat or non-cat with logistic regerssion. The main steps as follows:
## Load/explore/preprocessing dataset.
- data is stored in h5 format, using h5py to read.
- input X is the images with shape (m,n_height,n_width,n_channel), output Y is the classification (=1, is a cat, =0, not a cat), with shape(1,m).
- input X will be first reshaped to (n_h*n_w*n_c,m) and then standarized by '/255' to better fit with NN model.
## Define the model structure
![](datasets/LR.png)

with only 1 layer -- the output layer, and only 1 unit in that layer.

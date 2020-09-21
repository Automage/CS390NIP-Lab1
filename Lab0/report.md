Pranav Nair

nair51@purdue.edu

https://github.com/Automage/CS390NIP-Lab1

# Resources used:
Spring 2020 Lec 3: https://drive.google.com/file/d/11i4d1qqnABhxY84RRxiKPC83ENBUhCwE/view

Numpy tips for ReLU implementation: https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy

# Completed parts of the lab:
- Sigmoid and sigmoid derivative functions.
- Training function (used backpropagation properly).
    - Accuracy = 66%.
    - Used mini-batches of 64.
- Fully functioning 2-layer neural net. Added this model into the pipeline such that it is chosen when ALGORITHM = "custom_net".
- EC: implemented a second activation function (ReLU) that the network can use. The activation is specified in the constructor.

- Implemented a function that builds a 2 layer neural net using either Tensorflow or Keras. 
    - Accuracy = 97.39%

- The image data values are between 0 and 255. Preprocessed this such that values are between 0.0 and 1.0.
- Created an F1 score confusion matrix in the evaluation function. It should be printed along with accuracy.

# Summary:
Most of the work involving the custom neural network had to do with training. The back-propagation algorithm was understood and closely implemented like the lecture 2 slides presented. One hot encoding was also implemented in the predict() function. 

The Keras neural network was implemented by adding two layers to the keras model using keras.layers.Dense(). This was referenced from lecture 3 (including last semester's video). The optimizer and loss function used was Adam and categorical_crossentropy, respectively.

# Outputs
## Custom neural network
```
Testing Custom_NN.
Classifier algorithm: custom_net
Classifier accuracy: 66.280000%
F - scores:  [0.874 0.051 0.782 0.735 0.774 0.767 0.894 0.071   nan 0.688]
Confusion Matrix (x: actual, y: predicted):
[[  970.     4.    15.     5.     3.    11.    12.    90.   117.    13.   1240.]
 [    0.    30.     0.     0.     0.     0.     0.    16.     2.     0.   48.]
 [    0.   245.   984.    12.     2.     2.     3.   125.   113.     0.   1486.]
 [    3.   262.     9.   967.     1.    15.     0.   146.   206.    13.   1622.]
 [    0.   149.    11.     3.   949.    10.    11.   126.   172.    40.   1471.]
 [    2.   203.     0.    13.     1.   830.     5.    50.   163.     6.   1273.]
 [    5.    86.     8.     0.     6.    20.   926.     2.    57.     3.   1113.]
 [    0.     0.     0.     0.     0.     0.     0.    38.     1.     0.   39.]
 [    0.     0.     0.     0.     0.     0.     0.     3.     0.     0.   3.]
 [    0.   156.     5.    10.    20.     4.     1.   432.   143.   934.   1705.]
 [  980.  1135.  1032.  1010.   982.   892.   958.  1028.   974.  1009.   10000.]]
```
## Keras model
```
Testing TF_NN.
Classifier algorithm: tf_net
Classifier accuracy: 97.390000%
F - scores:  [0.983 0.989 0.975 0.965 0.971 0.97  0.979 0.972 0.97  0.962]
Confusion Matrix (x: actual, y: predicted):
[[  967.     0.     6.     0.     0.     2.     8.     1.     1.     2.   987.]
 [    0.  1124.     0.     0.     0.     0.     3.     6.     0.     5.   1138.]
 [    1.     3.  1005.     0.     4.     0.     1.    12.     3.     0.   1029.]
 [    3.     1.     4.  1001.     1.    24.     1.    10.     8.    11.   1064.]
 [    0.     0.     1.     1.   968.     1.     6.     5.     4.    25.   1011.]
 [    4.     1.     0.     0.     0.   857.     5.     0.     2.     6.   875.]
 [    1.     2.     2.     0.     2.     3.   929.     0.     1.     0.   940.]
 [    0.     1.     4.     2.     1.     0.     0.   984.     0.     5.   997.]
 [    2.     3.    10.     4.     2.     5.     5.     2.   953.     4.   990.]
 [    2.     0.     0.     2.     4.     0.     0.     8.     2.   951.   969.]
 [  980.  1135.  1032.  1010.   982.   892.   958.  1028.   974.  1009.   10000.]]
```

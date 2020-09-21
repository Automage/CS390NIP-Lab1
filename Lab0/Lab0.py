
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784
LAYER1_N = 512

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"

np.set_printoptions(precision=3, threshold=1500, suppress=True)


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, useReLU=False):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.activ = self.__sigmoid if not useReLU else self.__relu
        self.activDerivative = self.__sigmoidDerivative if not useReLU else self.__reluDerivative

    # Activation function.
    def __sigmoid(self, x):
        # print("sig shape ", x.shape)
        e = np.exp(-x)
        # print("x : ", x[0][0])
        # print("e^x : ", e[0][0])
        return 1 / (1 + e)

    # Activation prime function. Assumes sigmoid(x), not x, as input
    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    # ReLU activation
    def __relu(self, x):
        return np.maximum(x, 0)

    # ReLU activation derivative
    def __reluDerivative(self, x):
        return np.where(x <= 0, 0, 1)

    def __lossDerivative(self, x, y):
        return x - y

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield (i, l[i : i + n])

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 64):
        for j in range(epochs):
            print('EPOCH ', j)
            for index, inp in self.__batchGenerator(xVals, mbs):
                # print(f"EPOCH {j} | Minibatch {index}")
                yTrunc = yVals[index : index + mbs]
                
                # Forward pass
                l1Out, l2Out = self.__forward(inp)

                # Backprop
                l2e = self.__lossDerivative(l2Out, yTrunc)
                l2d = l2e * self.activDerivative(l2Out)
                l2a = np.dot(l1Out.T, l2d) * self.lr
                l1e = np.dot(l2d, self.W2.T)
                l1d = l1e * self.activDerivative(l1Out)
                l1a = np.dot(inp.T, l1d) * self.lr

                # Adjustments
                self.W2 -= l2a
                self.W1 -= l1a

    # Forward pass.
    def __forward(self, input):
        layer1 = self.activ(np.dot(input, self.W1))
        layer2 = self.activ(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        maxIndicies = np.argmax(layer2, axis=1)

        oneHot = np.zeros(layer2.shape)
        for i in range(maxIndicies.size):
            index = maxIndicies[i]
            oneHot[i][index] = 1

        return oneHot



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrainP, xTestP = xTrain / 255.0, xTest / 255.0
    xTrainP = xTrainP.reshape(60000, 28*28)
    xTestP = xTestP.reshape(10000, 28*28)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, LAYER1_N)
        model.train(xTrain, yTrain, 0)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = keras.Sequential()
        lossType = keras.losses.categorical_crossentropy
        # Version of backprop
        opt = tf.optimizers.Adam()
        
        # First layer
        model.add(keras.layers.Dense(LAYER1_N, input_shape=(IMAGE_SIZE,), activation=tf.nn.sigmoid))
        # Second layer
        model.add(keras.layers.Dense(NUM_CLASSES, input_shape=(LAYER1_N,), activation=tf.nn.sigmoid))
        model.compile(loss=lossType, optimizer=opt)

        # Train model
        model.fit(xTrain, yTrain, epochs=5)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        # Run model
        preds = model.predict(data)

        # One hot encoding
        maxIndicies = np.argmax(preds, axis=1)
        oneHot = np.zeros(preds.shape)
        for i in range(maxIndicies.size):
            index = maxIndicies[i]
            oneHot[i][index] = 1

        return oneHot
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    conf = np.zeros((NUM_CLASSES + 1,NUM_CLASSES + 1)) # + 1 for total column/row

    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        predIndex = np.argmax(preds[i])
        ansIndex = np.argmax(yTest[i])
        conf[predIndex][ansIndex] += 1

    conf[NUM_CLASSES] = conf.sum(axis=0)
    conf[:, NUM_CLASSES] = conf.sum(axis=1)

    fscores = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        precision = conf[i][i] / conf[i][NUM_CLASSES]
        recall = conf[i][i] / conf[NUM_CLASSES][i]
        fscores[i] = 2 * ((precision * recall) / (precision + recall))
    
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("F - scores: ", fscores)
    print('Confusion Matrix:')
    print(conf)




#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()

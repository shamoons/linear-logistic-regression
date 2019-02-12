# Task 7
from sklearn.datasets import load_iris
import numpy as np

X, Y = load_iris(return_X_y=True)


def sigmoid(z):
    return np.power(1 + np.exp(-z), -1)


def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # considering we have 4 nodes in the hidden layer
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        output_sigmoid_derivative = sigmoid_derivative(self.output)
        layer1_derivative = sigmoid_derivative(self.layer1)
        d_weights2 = np.dot(
            self.layer1.T, 2*(self.y - self.output) * output_sigmoid_derivative)
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output)
                                                 * output_sigmoid_derivative, self.weights2.T)*layer1_derivative)

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


Y = np.array(Y)
NN = NeuralNetwork(X, Y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(Y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        # mean sum squared loss
        print("Loss: \n" + str(np.mean(np.square(Y - NN.feedforward()))))
        print("\n")

    NN.train(X, Y)

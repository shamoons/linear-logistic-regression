import numpy as np

import pandas

def sigmoid(z):
  return 1/(1+ np.exp(-z))

def h(x, w):
  return sigmoid(np.dot(w.T, x))

def cost(X, W, Y):
  totalCost = 0
  for i in range(Y.size):
    diff = h(X[i], W) - Y[i]
    squared = diff * diff
    totalCost += squared
  
  return - totalCost / Y.size

alpha = 0.1
iris_data = np.loadtxt('Iris-binary.csv', delimiter=',')
inputX = iris_data[:,1:5]
outputY = iris_data[:,5]

normalizedX = np.ones((outputY.size, 5))
normalizedX[:,1] = inputX[:,0]
normalizedX[:,2] = inputX[:,1]
normalizedX[:,3] = inputX[:,2]
normalizedX[:,4] = inputX[:,3]

weights = np.ones((5,))
errorDiff = 100

for iter in range(1000):
  currentCost = cost(normalizedX, weights, outputY)
  if iter % 50 == 0:
    print(iter, 'iteration', weights)
    print('Cost', currentCost)
    print('Error', errorDiff)
  
  for i in range(outputY.size):
    errorDiff = h(normalizedX[i], weights) - outputY[i]
    weights[0] = weights[0] - (1 / outputY.size) * alpha * (errorDiff) * normalizedX[i][0]
    weights[1] = weights[1] - (1 / outputY.size) * alpha * (errorDiff) * normalizedX[i][1]
    weights[2] = weights[2] - (1 / outputY.size) * alpha * (errorDiff) * normalizedX[i][2]
    weights[3] = weights[3] - (1 / outputY.size) * alpha * (errorDiff) * normalizedX[i][3]
    weights[4] = weights[4] - (1 / outputY.size) * alpha * (errorDiff) * normalizedX[i][4]


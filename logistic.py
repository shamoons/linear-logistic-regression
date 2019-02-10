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

iris_data = np.loadtxt('Iris-binary.csv', delimiter=',')
inputX = iris_data[:,1:5]
outputY = iris_data[:,5]

normalizedX = np.ones((outputY.size, 5))
normalizedX[:,1] = inputX[:,0]
normalizedX[:,2] = inputX[:,1]
normalizedX[:,3] = inputX[:,2]
normalizedX[:,4] = inputX[:,3]

print(iris_data)
print(normalizedX)
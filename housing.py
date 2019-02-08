import numpy as np

import pandas

np.random.seed(1)

alpha = 0.1

def h(x, w):
  return np.dot(w.T, x)

def cost(X, W, Y):
  totalCost = 0
  for i in range(47):
    diff = h(X[i], W) - Y[i]
    squared = diff * diff
    totalCost += squared
  
  return totalCost / 2

housing_data = np.loadtxt('Housing.csv', delimiter=',')

x1 = housing_data[:,0]
x2 = housing_data[:,1]
y = housing_data[:,2]

avgX1 = np.mean(x1)
stdX1 = np.std(x1)
normX1 = (x1 - avgX1) / stdX1
print('avgX1', avgX1)
print('stdX1', stdX1)

avgX2 = np.mean(x2)
stdX2 = np.std(x2)
normX2 = (x2 - avgX2) / stdX2

print('avgX2', avgX2)
print('stdX2', stdX2)

normalizedX = np.ones((47, 3))

normalizedX[:,1] = normX1
normalizedX[:,2] = normX2

np.savetxt('normalizedX.csv', normalizedX)

weights = np.ones((3,))

for boom in range(100):
  currentCost = cost(normalizedX, weights, y)
  if boom % 1 == 0:
    print(boom, 'iteration', weights[0], weights[1], weights[2])
    print('Cost', currentCost)

  for i in range(47):
    errorDiff = h(normalizedX[i], weights) - y[i]
    weights[0] = weights[0] - alpha * (errorDiff) * normalizedX[i][0]
    weights[1] = weights[1] - alpha * (errorDiff) * normalizedX[i][1]
    weights[2] = weights[2] - alpha * (errorDiff) * normalizedX[i][2]
  
print(weights)

predictedX = [1, (2100 - avgX1) / stdX1, (3 - avgX2) / stdX2]
firstPrediction = np.array(predictedX)
print('firstPrediction', firstPrediction)
firstPrediction = h(firstPrediction, weights)
print(firstPrediction)
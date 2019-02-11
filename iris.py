import numpy as np

#Task 4

def sigmoid(z):
  return 1/( 1+ np.exp(-z))

def h(w, x):
  return sigmoid(np.dot(w.T, x))

def gradient(a, y, x):
  dZ = a - y

  return np.dot(x, dZ.T) / y.size

def predict(w, x):
  a = h(w,x)
  print(a)
  a[a < 0.5] = 0
  a[a >= 0.5] = 1

  return a

iris_data = np.loadtxt('Iris-binary.csv', delimiter=',')
alpha = 0.1
Y = iris_data[:,5]
X = np.ones((5, Y.size))
X[1,:] = iris_data[:,1]
X[2,:] = iris_data[:,2]
X[3,:] = iris_data[:,3]
X[4,:] = iris_data[:,4]

W = np.ones((5,1))

for i in range(1000):
  print('Iteration', i)
  a = h(W, X)
  dW = gradient(a, Y, X)
  W = W - dW * alpha
  
predictions = predict(W, X)

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# Task 5
dataset = np.loadtxt('Iris-binary.csv', delimiter=',')

Y = dataset[:,5]
X = dataset[:,1:4]
model = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(X, Y)

accuracy = model.score(X, Y)

print('Accuracy', accuracy)


# Task 6
from sklearn.datasets import load_iris
X, Y = load_iris(return_X_y=True)
model = LogisticRegression(random_state=0, solver='lbfgs',max_iter=500, multi_class='multinomial').fit(X, Y)
accuracy = model.score(X, Y)
print('Accuracy', accuracy)


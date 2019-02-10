import numpy
import matplotlib.pyplot as plot
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('Housing.csv', header=None)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

linearRegressor = LinearRegression()

xnorm = sklearn.preprocessing.scale(x)
scaleCoef = sklearn.preprocessing.StandardScaler().fit(x)
mean = scaleCoef.mean_
std = numpy.sqrt(scaleCoef.var_)

stuff = linearRegressor.fit(xnorm, y)

predictedX = [[(2100 - mean[0]) / std[0], (3 - mean[1]) / std[1]], [(1600 - mean[0]) / std[0], (2 - mean[1]) / std[1]]]
yPrediction = linearRegressor.predict(predictedX)
print('predictedX', predictedX)
print('predict', yPrediction)

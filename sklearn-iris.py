import numpy
import pandas
import sklearn
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pandas.read_csv('Iris-binary.csv', header=None)

x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

logisticRegression = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')

model = logisticRegression.fit(x, y)

print(model.score(x, y))
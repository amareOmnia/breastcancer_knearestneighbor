import numpy
from sklearn import preprocessing, neighbors
import pandas
from sklearn.model_selection import train_test_split

# access data file, replace ? fields, remove id col
dataFile = pandas.read_csv('data/breast-cancer-wisconsin.data')
dataFile.replace('?', -99999, inplace=True)  # outlier number disqualifies the data point
dataFile.drop(['id'], 1, inplace=True)

# form arrays of testing data
X = numpy.array(dataFile.drop(['class'], 1))
y = numpy.array(dataFile['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy: ', accuracy)

example_measures = numpy.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 1, 3, 4, 3, 2, 1]])
# Avoid deprication error through reshape, needs num of features^
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print('Prediction: ', prediction)

'''
TASKS:
- choice between random example or user input
- make it a method with .data file input variable, and optional user input'''
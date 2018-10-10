#Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#In this step we are going to take a look at the data a few different ways:

#Dimensions of the dataset.
#Peek at the data itself.
#Statistical summary of all attributes.
#Breakdown of the data by the class variable.

# Displays the dimensions of the data
print( "Data dimensions",)
print(dataset.shape, "\n")

# displays first 20 rows of data
print( "First rows of data")
print(dataset.head(20), "\n")

# Some basic measurements of the data set
print( "Data Summary")
print(dataset.describe(),"\n")

# class distribution
print("Class descriptions")
print(dataset.groupby('class').size(),"\n")

#We will split the loaded dataset into two, 80% of which we will use to train our models
# and 20% that we will hold back as a validation dataset.

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# Make predictions on validation dataset

SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print("Chosen Algorithm", name, "\n")
print(accuracy_score(Y_validation, predictions),"\n")
print(confusion_matrix(Y_validation, predictions), "\n")
print(classification_report(Y_validation, predictions))

import pickle
    with open("Hellosvm.pickle", "wb") as f:
  pickle.dump(SVM, f)



print( "New Data", "\n")
b = []




number = float(input("Type in sepal - length "))
b.append(number)

number1 = float(input("Type in sepal - width "))
b.append(number1)

number2 = float(input("Type in petal - length "))
b.append(number2)

number3 = float(input("Type in petal - width "))
b.append(number3)

print(b)

#for x in input("Please list data: ").split():
#    b.append(int(x))
Xnew = numpy.array([b])
print("X_new.shape: {}".format(Xnew.shape))
newprediction = SVM.predict(Xnew)

print("Prediction of Species: {}".format(newprediction))

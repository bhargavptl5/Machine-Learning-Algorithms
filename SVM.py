# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:04:22 2018

@author: Bhargav
"""

import numpy as mp
from sklearn import datasets,metrics
from sklearn.svm import SVC

iris=datasets.load_iris()

trainingData = iris.data[range(0,150,2),:]
trainingClass = iris.target[range(0,150,2)]

testData = iris.data[range(1,150,2),:]
testClass = iris.target[range(1,150,2)]

clf = SVC(kernel='linear')
clf.fit(trainingData,trainingClass)

predictions = clf.predict(testData)

#print(predictions)
print("\n\n-------Linear Kernel-------\n")
print("Accuracy :",metrics.accuracy_score(testClass,predictions)*100," %\n")

print("Classification Report :\n",metrics.classification_report(testClass,predictions))

print("Confusion Matrix :\n",metrics.confusion_matrix(testClass,predictions))

clf = SVC(kernel='poly')
clf.fit(trainingData,trainingClass)

predictions = clf.predict(testData)

#print(predictions)
print("\n\n-------Polynomial Kernel-------\n")
print("Accuracy :",metrics.accuracy_score(testClass,predictions)*100," %\n")

print("Classification Report :\n",metrics.classification_report(testClass,predictions))

print("Confusion Matrix :\n",metrics.confusion_matrix(testClass,predictions))

clf = SVC(kernel='rbf')
clf.fit(trainingData,trainingClass)

predictions = clf.predict(testData)

#print(predictions)
print("\n\n-------RBF Kernel-------\n")
print("Accuracy :",metrics.accuracy_score(testClass,predictions)*100," %\n")

print("Classification Report :\n",metrics.classification_report(testClass,predictions))

print("Confusion Matrix :\n",metrics.confusion_matrix(testClass,predictions))
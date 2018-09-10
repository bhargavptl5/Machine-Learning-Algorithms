# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:39:40 2018

@author: Bhargav
"""

import numpy as mp
from sklearn import linear_model,datasets,metrics

iris=datasets.load_iris()

trainingData = iris.data[range(0,150,2),:]
trainingClass = iris.target[range(0,150,2)]

testData = iris.data[range(1,150,2),:]
testClass = iris.target[range(1,150,2)]

clf = linear_model.LogisticRegression()
clf.fit(trainingData,trainingClass)

predictions = clf.predict(testData)

print(predictions)
print("\n\n-------Logistic Regression-------\n")
print("Accuracy :",metrics.accuracy_score(testClass,predictions)*100," %\n")

print("Classification Report :\n",metrics.classification_report(testClass,predictions))

print("Confusion Matrix :\n",metrics.confusion_matrix(testClass,predictions))


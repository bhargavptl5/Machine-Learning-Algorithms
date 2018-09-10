# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:36:19 2018

@author: Bhargav
"""

import numpy as mp
from sklearn import tree,datasets,metrics

iris=datasets.load_iris()

trainingData = iris.data[range(0,150,2),:]
trainingClass = iris.target[range(0,150,2)]

testData = iris.data[range(1,150,2),:]
testClass = iris.target[range(1,150,2)]

clf = tree.DecisionTreeClassifier()
clf.fit(trainingData,trainingClass)

predictions = clf.predict(testData)

#print(predictions)
print("\n\n-------Decision Tree ID3-------\n")
print("Accuracy :",metrics.accuracy_score(testClass,predictions)*100," %\n")

print("Classification Report :\n",metrics.classification_report(testClass,predictions))

print("Confusion Matrix :\n",metrics.confusion_matrix(testClass,predictions))
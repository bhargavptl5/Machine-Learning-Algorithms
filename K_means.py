# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:00:04 2018

@author: Bhargav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:39:40 2018

@author: Bhargav
"""

import numpy as mp
from sklearn import datasets,metrics
from sklearn.cluster import KMeans

iris=datasets.load_iris()

trainingData = iris.data[range(0,150,2),:]

testData = iris.data[range(1,150,2),:]


kmeans = KMeans(n_clusters=2, random_state=0).fit(trainingData)

predictions = kmeans.predict(testData)



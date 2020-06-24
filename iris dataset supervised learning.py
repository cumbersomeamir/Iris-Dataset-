# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:35:09 2020

@author: Amir
"""
'''Importing libraries which are required'''
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 

iris = load_iris() 

print("target names: {}".format(iris['target_names']))
print("Feature names: {}".format(iris['feature_names']))
print("Type of data: []".format(type(iris['data'])))
print("Shape of data: {}".format(iris['data'].shape))

print("Type of target: {}".format(type(iris['target']))) 
print("Shape of Target: {}".format(iris['target'].shape))
print("Target:\n{}".format(iris['target']))

print(iris['data'])

#Spllting the datatset 
X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'],random_state = 0)
#X_train = training data with features
#y_train = labels for the training dataset
#Printing the shapes of the training samples 
print("The shape of X_train is: {}".format(X_train.shape))
print("The shape of y_train is: {}".format(y_train.shape))
#X_test = testing data with features
#y_test = testing data labels 
#Printing the shape of the testing samples 
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

'''BUILDING THE MODEL'''
#Initialising the model
knn = KNeighborsClassifier(n_neighbors =1 )
#Fitting the model 
knn.fit(X_train,y_train)


'''Testing the model'''
#Creating a similar entry
new = np.array([[5,2.9,1,0.2]])
print("the shape of the newest array entry is: {}".format(new.shape))

#generating the prediction 
prediction = knn.predict(new)
print("Prediction:{}".format(prediction))
print("Predicted target name {}".format(iris['target_names'][prediction]))


'''Evalauting the model'''
y_pred = knn.predict(X_test)

print("Testing set predictions are: \n {}".format(y_pred))
print("Test set score (np.mean):{:.2f}".format(np.mean(y_pred == y_test)))
#Calculating the test set score(KNN score) 
score = knn.score(X_test,y_test)

#Expressing the test set score as a percentage 
print("{} %".format(score*100))



     

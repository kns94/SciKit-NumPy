#This code will describe my hands-on experience with Pythno Scikit
#Reference - Data School Videos on Youtube

#Data-set used: Iris supervised data

#Importing Iris data set
from sklearn.datasets import load_iris

iris=load_iris()
type(iris)

#Printing data
#print iris.data

#Printing feature names
#print(iris.feature_names)

#Print target - Integers representing the class label the data belongs to
#print(iris.target)

#Print target_names - Class labels
#print(iris.target_names)

#Check types of features and response
#print type(iris.data)
#print type(iris.target)

#Checking shape of features
#print(iris.data.shape)
#print(iris.target.shape)

#Store data in X
X=iris.data
#print(X)

#Store class label in Y
y=iris.target 
#print(y)



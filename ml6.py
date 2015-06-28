#This model will try different values for K in KNN and will find out which value works best
#Source-DataSchool

from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)

#range of possible values of k
k_range=range(1,26)
scores=[]

for k in k_range:
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred=knn.predict(X_test)
	scores.append(metrics.accuracy_score(y_test,y_pred))

#print(scores)

#Allow Matlab plots to appear inline
#matplotlib inline

#Plot relationship between k and testing accuracy
print(plt.plot(k_range,scores))
plt.xlabel('Value of K')
plt.xlabel('Testing Accuracy')

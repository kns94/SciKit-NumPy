#In this programm, I will prepare models to predict class labels for the iris data-set
#Source: Data School

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

iris=load_iris()

X=iris.data
y=iris.target

#Print Shapes of X and Y
#print(X.shape)
#print(y.shape) 

knn=KNeighborsClassifier(n_neighbors=5)
#print(knn)

#Train the data
knn.fit(X,y)	

#Predictor
print(knn.predict([3,5,4,2]))

X_new=[[3,5,4,2],[5,4,3,2]]
print(knn.predict(X_new))

logreg=LogisticRegression()
logreg.fit(X,y)

print(logreg.predict([3,5,4,2]))
print(logreg.predict(X_new))


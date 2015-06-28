#Evaluating which model works best using in this code
#Source: Data School

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
X=iris.data
y=iris.target

logreg=LogisticRegression()
logreg.fit(X,y)

#Predicting model
y_pred=logreg.predict(X)
#print(y_pred)
#print(len(y_pred))

#Accuracy Score
print(metrics.accuracy_score(y,y_pred))

knn5=KNeighborsClassifier(n_neighbors=5)
knn5.fit(X,y)
y_pred1=knn5.predict(X)
print(metrics.accuracy_score(y,y_pred1))

knn1=KNeighborsClassifier(n_neighbors=1)
knn1.fit(X,y)
y_pred2=knn1.predict(X)
print(metrics.accuracy_score(y,y_pred2))

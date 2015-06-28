#Model Evaluation part-2
#DataSchool

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)

#Print shapes
#print(X_train.shape)
#print(X_test.shape)

#print(y_train.shape)
#print(y_test.shape)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

knn5=KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train,y_train)
y_pred1=knn5.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred1))

knn1=KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train,y_train)
y_pred2=knn1.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred2))

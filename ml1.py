from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
from sklearn import datasets
from StringIO import StringIO
import csv

def main():
	#Creating training and test sets; skipping the header row with [1:]
	#dataset=np.genfromtxt('Data/Original.csv',delimiter='\t')
	#print(dataset)	

	#f=open("Data/Original.csv")
	#f.readline()
	#reader=csv.reader(f,delimiter="\t")
	
	#count=0
	#for data in reader:
	#	count=count+1

	#print(count)

	iris=np.genfromtxt('Data/iris1.csv',delimiter=',')
	#print(iris)
	target = [x[0] for x in iris]
	train = [x[0:] for x in iris]
	test=np.genfromtxt('Data/iris1.csv',delimiter=',')

	#Create and train random forest
	rf = RandomForestClassifier(n_estimators=100)
	rf.fit(train, target)
	predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
	
	np.savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()
	

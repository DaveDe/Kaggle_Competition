#After testing, the following adjustments can be made:
#Reduce categories of nominal values that have a frequency of x% (currently 5%)


import numpy as np
import pandas as pd
import csv
import codecs

def mergeCategory(feature,category):
	for index,item in enumerate(feature):
		if(item == category):
			feature[index] = 'new'
	return feature

def getFrequencyOfCategoryInFeature(category,feature):
	totalElements = 0
	matchingElements = 0
	for item in feature:
		if(item==category):
			matchingElements += 1
		totalElements += 1
	frequency = ((matchingElements/totalElements)*100)
	return frequency

#all nominal values that have under 5% frequency are merged to a new common nominal value 'new'
def removeCategories(feature):
	valueSet = set(list(feature))
	valueList = list(valueSet)
	for category in valueList:
		frequency = getFrequencyOfCategoryInFeature(category,feature)
		if(frequency < 5):
			feature = mergeCategory(feature,category)
	return feature

#reading data
with codecs.open('Data/train.csv','r',encoding='utf-8',errors='ignore') as f:
    readCSV = csv.reader(f, delimiter=',')
    data = list(readCSV)
data = np.array(data)

Y = data[1:,1].reshape(len(data)-1,1).astype(np.float)
binary_features = data[1:,10:]
nominal_features = data[1:,2:10]

#reduce the number of categories for each nominal feature
for index,column in enumerate(nominal_features.T):
	nominal_features.T[index] = removeCategories(column)

#convert all nominal features into dummy matricies, and append to training data
dummy_matrix = np.zeros(len(nominal_features.T[0]),dtype=np.int) #initialize with one column, which will later be removed
dummy_matrix = dummy_matrix.reshape(len(dummy_matrix),1)

for index,column in enumerate(nominal_features.T):
	panda_dummy_matrix = pd.get_dummies(nominal_features.T[index])
	np_matrix = np.asarray(panda_dummy_matrix)
	for col in np_matrix.T:
		col = col.reshape(len(col),1)
		dummy_matrix = np.hstack((dummy_matrix, col))

dummy_matrix = dummy_matrix[:,1:]

X = np.hstack((dummy_matrix,binary_features))
X = X.astype(np.int)

#at this point, X consists of all relevent features, broken down into binary
#Y consists of the associated value which we are trying to later predict

#Feature selection performed on X:

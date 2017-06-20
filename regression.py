#After testing, the following adjustments can be made:

#Reduce categories of nominal values that have a frequency of x% (currently 5%)
#Adjust the number of features PCA returns (currently 50)
#ANN: change activation function: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’} and # of nodes in hidden layer (currently 100)

import numpy as np
import pandas as pd
import csv
import codecs
from sklearn import decomposition #for pca
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

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

def convertNominalToBinary(binary_features,nominal_features):

	#reduce the number of categories for each nominal feature
	for index,column in enumerate(nominal_features.T):
		nominal_features.T[index] = removeCategories(column)

	#convert all nominal features into dummy matricies, and append to training data
	dummy_matrix = np.zeros(len(nominal_features.T[0]),dtype=np.int) #initialize with one column, which will later be removed
	dummy_matrix = dummy_matrix.reshape(len(dummy_matrix),1)

	#get dummy matrix from each column, and append all dummy matricies together
	for index,column in enumerate(nominal_features.T):
		panda_dummy_matrix = pd.get_dummies(nominal_features.T[index])
		np_matrix = np.asarray(panda_dummy_matrix)
		for col in np_matrix.T:
			col = col.reshape(len(col),1)
			dummy_matrix = np.hstack((dummy_matrix, col))

	dummy_matrix = dummy_matrix[:,1:] #remove first column, which was initialized to all 0's

	X = np.hstack((dummy_matrix,binary_features))
	X = X.astype(np.int)
	return X

#reading training data
with codecs.open('Data/train.csv','r',encoding='utf-8',errors='ignore') as f:
    readCSV = csv.reader(f, delimiter=',')
    data = list(readCSV)
data = np.array(data)

binary_features = data[1:,10:]
nominal_features = data[1:,2:10]
X_train = convertNominalToBinary(binary_features,nominal_features)
Y_train = data[1:,1].reshape(len(data)-1,1).astype(np.float)

#at this point, X consists of all relevent features, broken down into binary
#Y consists of the associated value which we are trying to later predict

#############################################  FEATURE SELECTION  ##############################################

#Feature selection performed on X:

#PCA
pca = decomposition.PCA(n_components=40)
pca.fit(X_train)
X_train = pca.transform(X_train)


#############################################  REGRESSION MODELS  ##############################################

#get test data
with codecs.open('Data/test.csv','r',encoding='utf-8',errors='ignore') as f:
    readCSV = csv.reader(f, delimiter=',')
    data = list(readCSV)
data = np.array(data)

#transform X_test just as we transformed X_train
binary_features = data[1:,9:]
nominal_features = data[1:,1:9]
X_test = convertNominalToBinary(binary_features,nominal_features)
pca.fit(X_test)
X_test = pca.transform(X_test)

#use linear or polynomial regression model
"""poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
model = LinearRegression()
model.fit(X_train,Y_train)
prediction = model.predict(X_test)"""

#ANN
"""model = MLPRegressor(hidden_layer_sizes=(100,),solver="lbfgs",activation="relu")
model.fit(X_train,Y_train)
prediction = model.predict(X_test)"""

model = ElasticNet(alpha=1, l1_ratio=0.7)
model.fit(X_train,Y_train)
prediction = model.predict(X_test)

#write submission file
submission_prediction = prediction.astype(str)
submission_prediction = submission_prediction.reshape(len(submission_prediction),1)
ID = data[1:,0]
ID = ID.reshape(len(ID),1)

submission = np.hstack((ID,submission_prediction))

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID','y'])
    writer.writerows(submission)
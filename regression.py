import numpy as np
import csv
import codecs

#reading data
with codecs.open('Data/train.csv','r',encoding='utf-8',errors='ignore') as f:
    readCSV = csv.reader(f, delimiter=',')
    data = list(readCSV)
data = np.array(data)

Y = data[1:,1].reshape(len(data)-1,1).astype(np.float)
X = data[1:,2:]
nominal_features = X[:,0:8]
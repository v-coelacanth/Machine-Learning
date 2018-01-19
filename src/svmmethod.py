import numpy
import numpy as np
import csv
import scipy
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from file_handling import *

# First we import the data for training and testing.
fft_feature_all = np.genfromtxt('../output/train.csv', delimiter=",")
testfeature = np.genfromtxt('../output/validate.csv', delimiter=",")

# Second we extract the data from the training file to get its features and classes.
feature1 = fft_feature_all[0:(fft_feature_all.shape[0]),0:(fft_feature_all.shape[1]-1)]
class1 = fft_feature_all[0:(fft_feature_all.shape[0]),(fft_feature_all.shape[1]-1)]

""" Convert the classes from reals to ints.  """
for k in range(0,class1.shape[0]):
    class1[k] = int(class1[k])

# Third we do the preprocessing for data.  We will scale all the data
# using a minmax scaler. Note: we transfomr the testfeatures using the
# same transformation derived from the test data.
scaler = preprocessing.StandardScaler().fit(feature1)
feature1 = scaler.transform(feature1)
testfeature = scaler.transform(testfeature)
"""
scaler = MinMaxScaler()
scaler.fit(feature1)
scaler.fit(testfeature)
feature1 = scaler.transform(feature1)
testfeature = scaler.transform(testfeature)
"""

# This step we begin to use different classifiers. We use support vector
# machine on this one.
# The function: SVC, KFold and preprocessing we use here is citated from SKlearn
# Note: we are using a polynomial kernel of degree 2.
clf = svm.SVC(kernel='poly', degree=2, coef0=1)
# Here we fit the training data
clf.fit(feature1, class1)

# Use 10-fold to validate our data
acc = np.zeros((10,1))
kf = KFold(n_splits=10)
k = 0
for train_index, test_index in kf.split(feature1):
    X_train, X_test = feature1[train_index], feature1[test_index]
    y_train, y_test = class1[train_index], class1[test_index]
    clf.fit(X_train,y_train)
    acc[k,0] = clf.score(X_test, y_test)
    k = k + 1
    testresult1 = np.zeros(X_test.shape[0])
    for i in range(0, X_test.shape[0]):
        testresult1[i] = clf.predict(X_test[i:i+1, :])

# Put the testing file into classifier to get the prediction
testresult = np.zeros((testfeature.shape[0], 1))
for i in range(0, testresult.shape[0]):
    testresult[i,0] = clf.predict(testfeature[i:i+1, :])


# This step is just for outputting the data into the standard format for Kaggle.
""" Get list of files in directory Code/something/validation with file
    extension .au.  """

testlist = get_files_in_directory("../something/validation")
testlist = clean_file_list(testlist, ".au")



musicgenre = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Finally, we put all the results into result.csv. The format is: id, class

with open('../output/result.csv', 'wb') as csvfile:
	title = ['id','class']
	csvwriter = csv.writer(csvfile, delimiter = ',')
	csvwriter.writerow(title)
	for j in range(100):
    		pos1 = testlist[j]
    		vorder = musicgenre[int(testresult[j,0])]
    		res = [pos1,vorder]
    		csvwriter.writerow(res)

import numpy
import numpy as np
import csv
import scipy
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from file_handling import *

# First we import the data for training and testing.
#
fft_feature_all = np.genfromtxt('../output/train.csv', delimiter=",")

testfeature = np.genfromtxt('../output/validate.csv', delimiter=",")

# Second we extract the data from the training file to get its features and classes.
feature1 = fft_feature_all[0:(fft_feature_all.shape[0]), 0:(fft_feature_all.shape[1]-1)]

class1 = fft_feature_all[0:(fft_feature_all.shape[0]), (fft_feature_all.shape[1]-1)]

for k in range(0,class1.shape[0]):
    class1[k] = int(class1[k])

# Third we do the preprocessing for data.


scaler = preprocessing.StandardScaler().fit(feature1)
feature1 = scaler.transform(feature1)
testfeature = scaler.transform(testfeature)


feature1mean = np.mean(feature1, axis=0)
for j in range(feature1.shape[1]):
    for i in range(feature1.shape[0]):
        feature1[i, j] = feature1[i, j] - feature1mean[j]

# This step we begin to use different classifiers. We use Logistic Regression on this one.
# The function: LogisticRegression, KFold and preprocessing we use here is citated from SKlearn
log_reg = LogisticRegression()
log_reg.fit(feature1, class1)

# Use 10-fold to validate our data
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(feature1):
    X_train, X_test = feature1[train_index], feature1[test_index]
    y_train, y_test = class1[train_index], class1[test_index]
    log_reg.fit(X_train, y_train)
    # print log_reg.score(X_test, y_test)

log_reg.fit(X_train, y_train)
testresult1 = np.zeros(X_test.shape[0])
for i in range(0, X_test.shape[0]):
    testresult1[i] = log_reg.predict(X_test[i:i+1, :])

#print confusion_matrix(y_test, testresult1, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Put the testing file into classifier to get the prediction
testresult = np.zeros((testfeature.shape[0], 1))
for i in range(0, testresult.shape[0]):
    testresult[i, 0] = log_reg.predict(testfeature[i:i+1, :])

# This step is just for outputting the data into the standard format for Kaggle.
testlist = get_files_in_directory("../something/validation")
testlist = clean_file_list(testlist, ".au")

musicgenre = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Finally, we put all the results into result.csv. The format is: id, class
with open('../output/result.csv', 'wb') as csvfile:
    title = ['id', 'class']
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(title)
    for j in range(100):
        pos1 = testlist[j]
        vorder = musicgenre[int(testresult[j, 0])]
        res = [pos1, vorder]
        csvwriter.writerow(res)

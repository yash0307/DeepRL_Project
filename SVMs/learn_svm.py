import sys,os
import random
import json
import numpy as np
from termcolor import colored
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn import preprocessing

amazon_features = np.load('features_from_amazon_to_amazon.npy')
dslr_features = np.load('features_from_amazon_to_dslr.npy')
webcam_features = np.load('features_from_amazon_to_webcam.npy')

#amazon_labels = 
#dslr_labels = 
#webcam_labels = 

print(amazon_features.shape)
print(dslr_features.shape)
print(webcam_features.shape)

num_amazon_images = amazon_features.shape[0]
num_dslr_images = dslr_features.shape[0]
num_webcam_images = webcam_features.shape[0]
rep_dim = amazon_features.shape[1]

# Domain Classifier SVM
X = np.zeros((num_amazon_images + num_dslr_images, rep_dim), dtype='float')
Y = np.zeros((num_amazon_images + num_dslr_images, 1), dtype='float')
for i in range(0, num_amazon_images):
	X[i,:] = amazon_features[i,:]
	Y[i,:] = 0
for i in range(num_amazon_images, num_amazon_images + num_dslr_images):
	X[i,:] = dslr_features[i-num_amazon_images,:]
	Y[i,:] = 1
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = svm.LinearSVC()
clf.fit(X_scaled, Y)
yy_ = clf.decision_function(X_scaled)
print(np.var(abs(yy_)))
AP = average_precision_score(Y, yy_)
print(AP)
joblib.dump(clf, 'amazon_0_dslr_1.pkl')

# Domain Classifier SVM
X = np.zeros((num_webcam_images + num_dslr_images, rep_dim), dtype='float')
Y = np.zeros((num_webcam_images + num_dslr_images, 1), dtype='float')
for i in range(0, num_dslr_images):
        X[i,:] = dslr_features[i,:]
        Y[i,:] = 0
for i in range(num_dslr_images, num_dslr_images + num_webcam_images):
        X[i,:] = webcam_features[i-num_dslr_images,:]
        Y[i,:] = 1
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = svm.LinearSVC()
clf.fit(X_scaled, Y)
yy_ = clf.decision_function(X_scaled)
print(np.var(abs(yy_)))
AP = average_precision_score(Y, yy_)
print(AP)
joblib.dump(clf, 'dslr_0_webcam_1.pkl')

# Domain Classifier SVM
X = np.zeros((num_amazon_images + num_webcam_images, rep_dim), dtype='float')
Y = np.zeros((num_amazon_images + num_webcam_images, 1), dtype='float')
for i in range(0, num_amazon_images):
        X[i,:] = amazon_features[i,:]
        Y[i,:] = 0
for i in range(num_amazon_images, num_amazon_images + num_webcam_images):
        X[i,:] = webcam_features[i-num_amazon_images,:]
        Y[i,:] = 1
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = svm.LinearSVC()
clf.fit(X_scaled, Y)
yy_ = clf.decision_function(X_scaled)
print(np.var(abs(yy_)))
AP = average_precision_score(Y, yy_)
print(AP)
joblib.dump(clf, 'amazon_0_webcam_1.pkl')

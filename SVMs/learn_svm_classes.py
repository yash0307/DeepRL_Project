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

amazon_labels = np.load('labels_amazon.npy')
dslr_labels = np.load('labels_dslr.npy')
webcam_labels = np.load('labels_webcam.npy')

print(amazon_features.shape)
print(dslr_features.shape)
print(webcam_features.shape)

train_type = 'amazon'
test_type = 'webcam'

if train_type == 'amazon':
	X = amazon_features
	Y = amazon_labels
elif train_type == 'dslr':
	X = dslr_features
	Y = dslr_labels
elif train_type == 'webcam':
	X = webcam_features
	Y = webcam_labels
if test_type == 'amazon':
	Z = amazon_features
	Z_ = amazon_labels
elif test_type == 'dslr':
	Z = dslr_features
	Z_ = dslr_labels
elif test_type == 'webcam':
	Z = webcam_features
	Z_ = webcam_labels

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = svm.LinearSVC(multi_class='ovr')
clf.fit(X_scaled, Y)

scaler = preprocessing.StandardScaler().fit(Z)
Z_scaled = scaler.transform(Z)
yy_ = clf.predict(Z_scaled)
for i in range(0, yy_.shape[0]):
        print(yy_[i])
AP = accuracy_score(Z_, yy_)
print(AP)

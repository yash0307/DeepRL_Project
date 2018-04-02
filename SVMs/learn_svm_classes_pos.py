import sys,os
import random
import json
import numpy as np
from termcolor import colored
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
from sklearn import preprocessing

amazon_features = np.load('features_from_amazon_to_amazon.npy')
dslr_features = np.load('features_from_amazon_to_dslr.npy')
webcam_features = np.load('features_from_amazon_to_webcam.npy')

rep_dim = amazon_features.shape[1]

amazon_labels = np.load('labels_amazon.npy')
dslr_labels = np.load('labels_dslr.npy')
webcam_labels = np.load('labels_webcam.npy')

print(amazon_features.shape)
print(dslr_features.shape)
print(webcam_features.shape)

train_type = 'amazon'
test_type = 'webcam'

domain_svm = joblib.load('./domain_SVMs/'+train_type+'_0_'+test_type+'_1.pkl')


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
#pos_num = range(100,800,50)
s_pos = abs(domain_svm.decision_function(Z_scaled))
pos_num = [794]
for given_num in pos_num:
	idxs = np.argpartition(s_pos, given_num)[:given_num]
	s_pos_X = np.zeros((len(idxs), rep_dim), dtype='float')
	for i in range(0, len(idxs)):
		s_pos_X[i,:] = Z[idxs[i],:]
	scaler = preprocessing.StandardScaler().fit(s_pos_X)
	s_pos_X_scaled = scaler.transform(s_pos_X)
	s_pos_Y = clf.predict(s_pos_X_scaled)
	s_pos_SVM = svm.LinearSVC(multi_class='ovr')
	s_pos_SVM.fit(s_pos_X_scaled, s_pos_Y)
	yy_ = s_pos_SVM.predict(Z_scaled)
	AP = accuracy_score(yy_, Z_)
	print('S_pos size: ' + str(given_num))
	print(AP)
	print('----------------')

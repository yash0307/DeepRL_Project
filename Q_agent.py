from __future__ import division

### git @ yash0307 ###
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
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import keras
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import random

##### Load Data #####
def data_loader(source_domain, target_domain, num_s_pos_samples, num_reward_samples, data_dir):
	domain_1_reps = np.load(data_dir+'features_from_'+source_domain+'_to_'+source_domain+'.npy')
	domain_2_reps = np.load(data_dir+'features_from_'+source_domain+'_to_'+target_domain+'.npy')
	domain_1_labels = np.load(data_dir+'labels_'+source_domain+'.npy')
	domain_2_labels = np.load(data_dir+'labels_'+target_domain+'.npy')
	num_domain_1_images = domain_1_reps.shape[0]
	num_domain_2_images = domain_2_reps.shape[0]
	rep_dim = domain_1_reps.shape[1]
	if rep_dim != domain_2_reps.shape[1]:
		print('Something wrong with representations !!')
		sys.exit(1)
	return domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim
##### End data loader #####

##### Initialize S_pos_set and R_set #####
def gen_s_pos_and_reward_set(domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim):
	domain_class_data_X = np.concatenate((domain_1_reps, domain_2_reps), axis=0)
	domain_class_data_Y_1 = np.zeros((num_domain_1_images,1), dtype='float')
	domain_class_data_Y_2 = np.ones((num_domain_2_images,1), dtype='float')
	domain_class_data_Y = np.concatenate((domain_class_data_Y_1, domain_class_data_Y_2), axis=0).flatten()

	domain_class_scalar = preprocessing.StandardScaler().fit(domain_class_data_X)
	X_scaled_domain_class = domain_class_scalar.transform(domain_class_data_X)
	domain_classifier = svm.LinearSVC()
	domain_classifier.fit(X_scaled_domain_class, domain_class_data_Y)
	
	domain_class_2 = preprocessing.StandardScaler().fit(domain_2_reps)
	X_scaled_domain_2 = domain_class_2.transform(domain_2_reps)
	sample_distances = abs(domain_classifier.decision_function(X_scaled_domain_2))
	
	class_dist_dict = {}
	min_class_idx = int(min(domain_2_labels))
	max_class_idx = int(max(domain_2_labels))
	for given_class in range(min_class_idx, max_class_idx):
		sample_class_idxs = np.where(domain_2_labels == given_class)
		if given_class not in class_dist_dict.keys():
			class_dist_dict[given_class] = []
		for given_sample in sample_class_idxs[0]:
			given_sample_distance = sample_distances[given_sample]
			class_dist_dict[given_class].append((given_sample, given_sample_distance))
	
	reward_set = {}
	for given_class in class_dist_dict.keys():
		if given_class not in reward_set.keys():
			reward_set[given_class] = []
		given_class_list = sorted(class_dist_dict[given_class], key=lambda x:x[1])
		for given_sample in given_class_list[:num_reward_samples]:
			reward_set[given_class].append(given_sample[0])

	s_pos = []
	sample_distances_pairs = []
	for given_sample in range(0, len(sample_distances)):
		sample_distances_pairs.append((given_sample, sample_distances[given_sample]))

	sorted_sample_dist = sorted(sample_distances_pairs, key=lambda x:x[1])
	for given_sample in sorted_sample_dist[len(sorted_sample_dist)-num_s_pos_samples:]:
		s_pos.append(given_sample[0])
	return s_pos, reward_set
##### End Initialization #####

##### Train n-way classifier #####
def train_SVM(s_pos, domain_2_labels, domain_2_reps, rep_dim):
	num_samples = len(s_pos)
	X = np.zeros((num_samples, rep_dim), dtype='float')
	Y = np.zeros((num_samples, 1), dtype='float')
	for i in range(0, num_samples):
		given_sample = s_pos[i]
		X[i,:] = domain_2_reps[given_sample,:]
		Y[i] = domain_2_labels[given_sample]
	scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        given_SVM = svm.LinearSVC(multi_class='ovr')
        given_SVM.fit(X_scaled, Y)
	return given_SVM

def check_accu(reward_set, domain_2_reps, given_SVM, rep_dim):
	num_samples = sum([len(reward_set[i]) for i in reward_set.keys()])
	X = np.zeros((num_samples, rep_dim), dtype='float')
	Y = np.zeros((num_samples, 1), dtype='float')
	idx = 0
	for given_class in reward_set.keys():
		list_class = reward_set[given_class]
		for given_sample in list_class:
			X[idx,:] = domain_2_reps[given_sample,:]
			Y[idx] = given_class
			idx += 1
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	YY_ = given_SVM.predict(X_scaled)
	Y = Y.flatten()
	accu = accuracy_score(Y, YY_)
	return accu

##### End Train n-way classifier #####

##### Unlabeled predictions #####
def get_unlabelled_predictions(domain_1_reps, domain_2_reps, domain_1_labels):
	X = domain_1_reps
	Y = domain_1_labels
	Z = domain_2_reps
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	SVM = svm.LinearSVC(multi_class='ovr')
	SVM.fit(X_scaled, Y)
	scaler = preprocessing.StandardScaler().fit(Z)
	Z_scaled = scaler.transform(Z)
	unsupervised_labels = SVM.predict(Z_scaled)
	return unsupervised_labels
##### End Unlabeled predictions #####

def gen_state(SVM, domain_2_reps):
	X = domain_2_reps
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	hist_all = SVM.decision_function(X_scaled)
	return hist_all

if __name__ == '__main__':
	source_domain = 'amazon'
	target_domain = 'dslr'
	num_s_pos_samples = 100
	num_reward_samples = 3
	max_iters = 2000
	data_dir = '/home/yash/Sem2/DeepRL/Project/Amazon-finetune-features/'

	# Load the data
       	domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim = data_loader(source_domain, target_domain, num_s_pos_samples, num_reward_samples, data_dir)

	# Get positive and reward sets
	s_pos, reward_set = gen_s_pos_and_reward_set(domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim)

	domain_2_labels = get_unlabelled_predictions(domain_1_reps, domain_2_reps, domain_1_labels)
	
	for given_iter in range(0, max_iters):
		given_SVM = train_SVM(s_pos, domain_2_labels, domain_2_reps, rep_dim)
		given_accu = check_accu(reward_set, domain_2_reps, given_SVM, rep_dim)
		given_state = gen_state(given_SVM, domain_2_reps)
		print(given_accu)

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
def gen_s_pos_and_reward_set(domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim, dict_domain_2):
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
			dict_domain_2[given_sample[0]] = int(1)
	s_pos = []
	sample_distances_pairs = []
	for given_sample in range(0, len(sample_distances)):
		sample_distances_pairs.append((given_sample, sample_distances[given_sample]))

	sorted_sample_dist = sorted(sample_distances_pairs, key=lambda x:x[1])
	for given_sample in sorted_sample_dist[len(sorted_sample_dist)-num_s_pos_samples:]:
		s_pos.append(given_sample[0])
		dict_domain_2[given_sample[0]] = int(1)
	return s_pos, reward_set, dict_domain_2
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

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def gen_state_pos(SVM, s_pos, domain_2_reps, rep_dim, num_hist=10):
	num_samples = len(s_pos)
	num_classes = 31
	hist_out = np.zeros((num_classes, num_hist), dtype='float')
	X = np.zeros((num_samples, rep_dim), dtype='float')
	idx = 0
	for given_sample in s_pos:
		X[idx,:] = domain_2_reps[given_sample,:]
		idx += 1
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	hist_all = SVM.decision_function(X_scaled)
	svm_predictions = SVM.predict(X_scaled)
	pred_classes = list(set([int(i) for i in svm_predictions]))
	for given_sample_idx in range(0, len(s_pos)):
		given_sample = s_pos[given_sample_idx]
		given_rep = softmax(abs(hist_all[given_sample_idx]))
		given_sample_class = int(svm_predictions[given_sample_idx])
		class_idx = pred_classes.index(given_sample_class)
		given_val = given_rep[class_idx]
		hist_div = float(1)/float(num_hist)
		given_hist = int(float(given_val)/float(hist_div))
		hist_out[given_sample_class][given_hist] += 1
	return hist_out

def init_dict_domain_2(domain_2_reps, domain_2_labels):
	dict_domain_2 = {}
	for i in range(0, domain_2_reps.shape[0]): dict_domain_2[i] = int(0)
	return dict_domain_2	

def sample_images(domain_2_reps, domain_2_labels, dict_domain_2, rep_dim, sample_num=100):
	available_samples = [i for i in dict_domain_2.keys() if dict_domain_2[i] == 0]
	sampled_idxs = np.random.randint(low=0, high=len(available_samples), size=sample_num)
	given_reps = np.zeros((sample_num, rep_dim), dtype='float')
	given_labels = np.zeros((sample_num, 1), dtype='float')
	i = 0
	out_idxs = []
	for idx in sampled_idxs:
		given_idx = available_samples[idx]
		out_idxs.append(given_idx)
		given_reps[i,:] = domain_2_reps[given_idx,:]
		given_labels[i] = domain_2_labels[given_idx]
	return given_reps, given_labels, out_idxs

def gen_state_samples(sampled_reps, sampled_labels, sampled_idxs, SVM):
	
if __name__ == '__main__':
	source_domain = 'amazon'
	target_domain = 'dslr'
	num_s_pos_samples = 100
	num_reward_samples = 3
	max_iters = 2000
	sample_num = 100
	data_dir = '/home/yash/Sem2/DeepRL/Project/Amazon-finetune-features/'

	# Load the data
       	domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim = data_loader(source_domain, target_domain, num_s_pos_samples, num_reward_samples, data_dir)

        # Create a dict for samples in domain-2
        dict_domain_2 = init_dict_domain_2(domain_2_reps, domain_2_labels) # All indexes are zero by default

	# Get positive and reward sets
	s_pos, reward_set, dict_domain_2 = gen_s_pos_and_reward_set(domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim, dict_domain_2)

	# Get labels for domain-2 using classifier trained on domain-1
	domain_2_labels = get_unlabelled_predictions(domain_1_reps, domain_2_reps, domain_1_labels)

	# Sample a given number of data points from available samples	
	sampled_reps, sampled_labels, sampled_idxs = sample_images(domain_2_reps, domain_2_labels, dict_domain_2, rep_dim, sample_num)

	for given_iter in range(0, max_iters):
		given_SVM = train_SVM(s_pos, domain_2_labels, domain_2_reps, rep_dim)
		given_accu = check_accu(reward_set, domain_2_reps, given_SVM, rep_dim)
		h_pos = gen_state_pos(given_SVM, s_pos, domain_2_reps, rep_dim)
		print(h_pos)
		print(sum(sum(h_pos)))
		sys.exit(1)
		print(given_accu)

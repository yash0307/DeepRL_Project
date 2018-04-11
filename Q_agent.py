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
	
	s_pos = {}
	reward_set = {}
	for given_class in class_dist_dict.keys():
		if given_class not in s_pos.keys():
			s_pos[given_class] = []
			reward_set[given_class] = []
		given_class_list = sorted(class_dist_dict[given_class], key=lambda x:x[1])
		for given_sample in given_class_list[:num_reward_samples]:
			reward_set[given_class].append(given_sample[0])
		for given_sample in given_class_list[len(given_class_list)-num_s_pos_samples:]:
			s_pos[given_class].append(given_sample[0])
	return s_pos, reward_set
##### End Initialization #####

##### Train n-way classifier #####
##### End Train n-way classifier #####

##### Make State Representation #####
##### End Make State Representation #####

if __name__ == '__main__':
	source_domain = 'amazon'
	target_domain = 'dslr'
	num_s_pos_samples = 6
	num_reward_samples = 3
	data_dir = '/home/yash/Sem2/DeepRL/Project/Amazon-finetune-features/'

	# Load the data
       	domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim = data_loader(source_domain, target_domain, num_s_pos_samples, num_reward_samples, data_dir)

	# Get positive and reward sets
	s_pos, reward_set = gen_s_pos_and_reward_set(domain_1_reps, domain_2_reps, domain_1_labels, domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim)

	# Generate initial state representations

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


def test_check_accu(domain_2_reps, domain_2_labels, given_SVM, rep_dim):
        num_samples = domain_2_reps.shape[0]
        X = np.zeros((num_samples, rep_dim), dtype='float')
        Y = np.zeros((num_samples, 1), dtype='float')
        idx = 0
        for given_sample in range(0, num_samples):
        	X[idx,:] = domain_2_reps[given_sample,:]
                Y[idx] = domain_2_labels[given_sample]
                idx += 1
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        YY_ = given_SVM.predict(X_scaled)
        Y = Y.flatten()
        accu = accuracy_score(Y, YY_)
        return accu

if __name__ == '__main__':

	source_domain = 'amazon'
	target_domain = 'dslr'
	num_s_pos_samples = 100
	num_reward_samples = 2
	max_iters = 20000
	sample_num = 20
	max_explore_iter = 2000
	num_hist = 10
	num_classes = 31
	data_dir = '/home/yash/Project/Amazon-finetune-features/'
	update_point = 10
	svm_model = 'svm_best.pkl'
	given_SVM = joblib.load(svm_model)
       	domain_1_reps, domain_2_reps, domain_1_labels, test_domain_2_labels, num_domain_1_images, num_domain_2_images, rep_dim = data_loader(source_domain, target_domain, num_s_pos_samples, num_reward_samples, data_dir)
        accu = test_check_accu(domain_2_reps, test_domain_2_labels, given_SVM, rep_dim)
	print(accu)

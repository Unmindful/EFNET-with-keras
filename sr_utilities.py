from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import re, math
import os, glob, sys, threading, random
import scipy.io
from scipy import ndimage, misc
from skimage import filters, feature
from keras import backend as K
from keras import optimizers, regularizers
from keras.losses import mean_squared_error, mean_absolute_error 
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import SeparableConv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add, Conv2D, concatenate, Dropout, Lambda, Conv2DTranspose, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from subpixel_old import Subpixel

#Setting the parameter. Can handle once scale at a time
scale = 4
IMG_SIZE = (192/scale, 192/scale, 3)
BATCH_SIZE = 16
EPOCHS = 100
ResBlock = 4
MulBlock = 4
TRAIN_SCALES = [scale]
VALID_SCALES = [scale]


#Function to Generate Matlab File List
def get_image_list(data_path, scales=[2, 3, 4]):
    
	l = glob.glob(os.path.join(data_path,"*"))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l:
		if os.path.exists(f):	
			for i in range(len(scales)):
				scale = scales[i]
				string_scale = "_" + str(scale) + ".mat"
				if os.path.exists(f[:-4]+string_scale): 
					train_list.append([f, f[:-4]+string_scale])
	return train_list

##Function to Generate Image Batch List
def get_image_batch(train_list, offset):
    
	target_list = train_list[offset:offset+BATCH_SIZE]
	input_list = []
	gt_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, np.size(input_img, 0),np.size(input_img, 1), IMG_SIZE[2]])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, np.size(gt_img, 0), np.size(gt_img, 1), IMG_SIZE[2]])
	return input_list, gt_list


#Function to Generate Image Batch
def image_gen(target_list):
    
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset)
			yield (batch_x, [batch_y, batch_y, batch_y])


def PSNR(y_true, y_pred):
    
	return K.expand_dims(tf.image.psnr(y_true, y_pred, max_val=1.0),0)

def SSIM(y_true, y_pred):
    
	return K.expand_dims(tf.image.ssim(y_true, y_pred, max_val=1.0),0)

def sob_loss_function(y_true, y_pred):
    
	sob_t = tf.image.sobel_edges(y_true)
	sob_t = K.flatten(sob_t)
	sob_p = K.flatten(y_pred)
	loss = K.mean(K.abs(sob_p - sob_t))
	return loss

def sobel_loss():
    
	def my_loss(y_true, y_pred):
		return sob_loss_function(y_true, y_pred)
	return my_loss

def lap_loss_function(y_true, y_pred):
    
	lap_t = laplacian(y_true)
	lap_t = K.flatten(lap_t)
	lap_p = K.flatten(y_pred)
	loss = K.mean(K.abs(lap_p - lap_t))
	return loss

def laplacian_loss():
    
	def my_loss(y_true, y_pred):
		return lap_loss_function(y_true, y_pred)
	return my_loss

def step_decay(epoch):
    
	initial_lrate = 10**(-4)
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def bicubic_up(tensor):
    
	result_up = tf.image.resize_images(tensor, [IMG_SIZE[0]*scale, IMG_SIZE[1]*scale])
	return result_up

def bicubic_down(tensor):
    
	result_down = tf.image.resize_images(tensor, [IMG_SIZE[0], IMG_SIZE[1]])
	return result_down

def sobel(tensor):
    
	result_sobel = tf.image.sobel_edges(tensor)
	bsize, a, b, c, d = result_sobel.get_shape().as_list()
	bsize = K.shape(result_sobel)[0]
	result_sobel1 = K.reshape(result_sobel,[bsize, a, b, c*d])
	return result_sobel1

def laplacian(tensor):
    
	dx, dy = tf.image.image_gradients(tensor)
	d2x, d2x0 = tf.image.image_gradients(dx)
	d2y0, d2y = tf.image.image_gradients(dy)
	result_laplacian = add([d2x,d2y])
	return result_laplacian

#Recursive Residual Block
def RES_add_Block(x, res_number=1):
    
	model_input = Input(shape=(None, None, 64))
	model_conv_1 = SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(model_input)
	model_conv_2 = SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(model_conv_1)
	model_conv_3 = SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(model_conv_2)
	model = concatenate([model_input, model_conv_1, model_conv_2, model_conv_3])
	model_conv_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='he_uniform')(model)
	model_RDB_add = Model(model_input, model_conv_1x1)
	x_con = x
	for res in range(res_number):
		x = model_RDB_add(x)
		x_con = concatenate([x_con,x])
	x_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(x_con)
	x = x_1x1
	return x

#Non Recursive Shallow Block for Spatial Attention
def RES_mul_Block(x, resblock_number=1):
    
	model_input = Input(shape=(None, None, 64))
	model_conv_1 = SeparableConv2D(64, (1, 9), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_input)
	model_conv_2 = SeparableConv2D(64, (9, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_conv_1)
	model_conv_3 = SeparableConv2D(64, (1, 9), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_conv_2)
	model_conv_4 = SeparableConv2D(64, (9, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_conv_2)
	model = add([model_input, model_conv_4])
	model_insideres_mul = Model(model_input, model)
	for res in range(resblock_number):
		x = model_insideres_mul(x)
	return x

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
from sr_utilities import scale, ResBlock, MulBlock, TRAIN_SCALE, VALID_SCALE, get_image_list, image_gen, sobel, laplacian, PSNR, SSIM, sob_loss_function,lap_loss_function, step_decay, RES_add_Block, RES_mul_Block, sobel_loss, laplacian_loss

#Get the testing data

data_path = './Validation Data/Test Images/Urban100_4scale_rgb/'
path = './Predicted_image/Urban100_EFNet+_4x'
os.mkdir(path)
scale = 4
count = 0
time_c = 0

#As the image dimension is variable, for Efficient Subpixel Convolution we need dynamic dimension.
#BY means of for loop, image dimension changes dynamically

for im_name in os.listdir(data_path):
	img = image.load_img(data_path+im_name)
	row=np.size(img, 0)
	col=np.size(img, 1)
	channel=np.size(img,2)
	IMG_SIZE = (row, col, channel)
	BATCH_SIZE = 1
	TRAIN_SCALES = [scale]
	VALID_SCALES = [scale]

    input_img = Input(shape=IMG_SIZE)
    model_sob = Lambda(sobel)(input_img)
    model_lap = Lambda(laplacian)(input_img)
    model_feature1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(input_img)
        model_feature2  =concatenate([model_feature1,model_sob,model_lap])
    model_feature2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(model_feature2)

    model1 = model_feature2
    model2 = model_feature2
    model_con1 = model_feature2
    model_con2 = model_feature2

    for blocks in range (ResBlock):
        x1 = model1
        model1 = RES_add_Block(model1, res_number=4)
        model_con1 = concatenate([model_con1,model1])

    model_edge_sob = Conv2D(6, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_con1)
    model_edge_lap = Conv2D(3, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_con1)


    for blocks in range (MulBlock):
        model2 = RES_mul_Block(model2)
        model2 = concatenate([model_con2,model2])

    
    model1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='glorot_uniform')(model_con1)
    model1 = SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(model1)
    model1 = add([model1, model_feature1])
    model2 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model2)

#Upscaling the inputs by means of Efficient Subpixel Convolution
    if scale >= 2 and scale != 3:
        model1 = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(model1)
        model2 = Subpixel(1, (3,3), r = 2,padding='same',activation='relu')(model2)
        model_edge_sob = Subpixel(6, (3,3), r = 2,padding='same',activation='relu')(model_edge_sob)
        model_edge_lap = Subpixel(3, (3,3), r = 2,padding='same',activation='relu')(model_edge_lap)
        model_in = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(input_img)
    
    if scale >= 4:
        model1 = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(model1)
        model2 = Subpixel(1, (3,3), r = 2,padding='same',activation='relu')(model2)
        model_edge_sob = Subpixel(6, (3,3), r = 2,padding='same',activation='relu')(model_edge_sob)
        model_edge_lap = Subpixel(3, (3,3), r = 2,padding='same',activation='relu')(model_edge_lap)
        model_in = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(input_img)
    
    if scale >= 8:
        model1 = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(model1)
        model2 = Subpixel(1, (3,3), r = 2,padding='same',activation='relu')(model2)
        model_edge_sob = Subpixel(6, (3,3), r = 2,padding='same',activation='relu')(model_edge_sob)
        model_edge_lap = Subpixel(3, (3,3), r = 2,padding='same',activation='relu')(model_edge_lap)
        model_in = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(input_img)
    
    if scale == 3:
        model1 = Subpixel(64, (3,3), r = 3,padding='same',activation='relu')(model1)
        model2 = Subpixel(1, (3,3), r = 3,padding='same',activation='relu')(model2)
        model_edge_sob = Subpixel(6, (3,3), r = 3,padding='same',activation='relu')(model_edge_sob)
        model_edge_lap = Subpixel(3, (3,3), r = 3,padding='same',activation='relu')(model_edge_lap)
        model_in = Subpixel(64, (3,3), r = 3,padding='same',activation='relu')(input_img)


#Apply 1x1 convolution for Soblel and Laplacian edge loss purpose
    model_edge_sob = Lambda(lambda x:x, name="sobel")(model_edge_sob)
    model_edge_sob_1 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_edge_sob)

    model_edge_lap = Lambda(lambda x:x, name="lap")(model_edge_lap)
    model_edge_lap_1 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(model_edge_lap)

#Apply Hard Sigmoid for Spatial Attention purpose
    model2 = Activation('hard_sigmoid')(model2)


    model_sob_lap = add([model1, model_edge_sob_1, model_edge_lap_1])
    model_attention = multiply([model1, model2])
    model = add([model_attention, model_in, model_edge_sob_1, model_edge_lap_1])

#Reconstruction Block
    output_img = Conv2D(3, (3, 3), padding='same', kernel_initializer='glorot_uniform',name='4x_output')(model)
    output_img = Lambda(lambda x:x, name="o_f")(output_img)

    model = Model(inputs=[input_img], outputs=[output_img, model_edge_sob, model_edge_lap])
	model.load_weights('./Saved Models/sr_EFNet+_with100epoch_model.h5')
	adam = Adam(lr=0, epsilon=10**(-8))
	sgd = SGD(lr=0, momentum=0.9, decay=1e-4, nesterov=False, clipnorm=1)
	model.compile(adam, loss='mae', metrics=[PSNR,SSIM])

#Actual Testing begins here
	x = image.img_to_array(img)
	x = x.astype('float32') / 255
	x = np.expand_dims(x, axis=0)
	start_t = time.time()
	pred,a,b = model.predict(x)
	end_t = time.time()
	print ("end_t:",end_t,"start_t:",start_t)
	time_c=time_c+end_t-start_t
	print ("Time Consumption:",end_t-start_t)
	test_img = np.reshape(pred, (row*scale, col*scale,channel))
	print(np.size(test_img))
	imsave(path+'/im'+str(count)+'.png', test_img)
	count += 1
	print ("Image Number:",count)
print ("Total Time:",time_c)


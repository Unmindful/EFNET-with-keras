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
from sr_utilities import scale, IMG_SIZE, BATCH_SIZE, EPOCHS, TRAIN_SCALE, VALID_SCALE, get_image_list, image_gen, sobel, laplacian, PSNR, SSIM, sob_loss_function,lap_loss_function, step_decay, RES_add_Block, RES_mul_Block, sobel_loss, laplacian_loss


# Get the training and testing data
train_list = get_image_list("./Train Data/train_DIV2K_192x192rgb/", scales=TRAIN_SCALES)
test_list = get_image_list("./Validation Data/val_Set14_192x192rgb/", scales=VALID_SCALES)

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
#model.load_weights('./Saved Models/sr_EFNet+_with100epoch_model.h5')
adam = Adam(lr=0, epsilon=10**(-8))
sgd = SGD(lr=0, momentum=0.9, decay=1e-4, nesterov=False, clipnorm=1)
model.compile(adam, loss= ['mae', sobel_loss(), laplacian_loss()], loss_weights = [.7, 0.15, 0.15], metrics={'o_f':[PSNR,SSIM]})
model.summary()
filepath="./saved_weights/weights-improvement-{epoch:02d}-{val_o_f_PSNR:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='o_f_PSNR', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
earlystopper = EarlyStopping(monitor='o_f_PSNR', patience=15, verbose=1, mode='max')
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, checkpoint, earlystopper]


print("Started training")
history = model.fit_generator(image_gen(train_list), steps_per_epoch=len(train_list) // BATCH_SIZE,  \
					validation_data=image_gen(test_list), validation_steps=len(test_list) // BATCH_SIZE,
					epochs=EPOCHS, workers=8, callbacks=callbacks_list, verbose=1)

print("Done training!!!")
print("Saving the final model ...")

#Creates a HDF5 file 
model.save('./Saved Models/sr_EFNet+_with100epoch_model.h5')  

#Deletes the existing model
del model  

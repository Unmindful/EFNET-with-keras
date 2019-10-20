# Single Image Super Resolution by Efficient Edge Fusion Network(Keras Implementation)

## Overview

Upscaling the image by means of edge fusion and spatial attention technique preserve the high level features. Here the technique is implemented with Keras on top of TensorFlow. Project work is done by Dr. Dimitrious Androutsos and myself on September 2019 in Ryerson University.

## Files

* Train.py: Runs the script to start training.
* Test.py: Upscales the images in a folder and saves in the given directory. Watch the scaling factor.
* subpixel_old.py: Contains the funcion for Efficient Subpixel Convolution; an upscaling technique.
* sr_utilities.py: The utility functions are kept in this script. Variables are to be changed here for other scaling factors and different sized tarining patches.
* PSNR_SSIM.m: Calculates the average PSNR & SSIM in a folder of images.

## Data

Matlab files are used as tarining and validation data. 
* aug_train_data.m: Flips, rotates the training data according to the desired patch
* aug_test_data.m: Generates the validation patch of given size
* downscale_testimage.m: Downscales the input as per given scale. It is used for the test purpose.
The training and validation mat files can be found [here](https://drive.google.com/file/d/1ug-B6FPuWfFKLays91rGUBuyrc9tHJD6/view?usp=sharing).
In addition, all the benchmark training and testing images is given [here](https://drive.google.com/file/d/1ug-B6FPuWfFKLays91rGUBuyrc9tHJD6/view?usp=sharing).  

## Implementation

Just simply run the training script by mentioning the scaling factor, and input training/validation data. For testing keep in mind the directory of the saved model. This technique doesn't support multiscale training. So, the training script should be run seperately for different scaling factor by changing the saved model name. 

#!/usr/bin/env python
# coding: utf-8


import numpy as np
# import tensorflow as tf
from PIL import Image
from tensorflow import keras
# from tensorflow.keras.preprocessing import image as kp_image
import matplotlib.pyplot as plt

def loadimg(path_to_img):
    """
    load the image and scale it to a proper size
    """
    img = Image.open(path_to_img)
    longer_dim = max(img.size)
    max_dim = 500 # the maximum size of the output image longer size
    scale = max_dim / longer_dim
    img = img.resize((round(img.size[0] * scale),round(img.size[1] * scale)),Image.ANTIALIAS)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    return img

def showimg(img):
    out = np.squeeze(img,axis = 0)
    out = out.astype('uint8')
    plt.axis('off')
    plt.imshow(out)


def pre_process_img(path_to_img):
    img = loadimg(path_to_img)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    img = processed_img.copy()
    if len(img.shape) == 4:
        img = np.squeeze(img,0)
    assert len(img.shape) == 3,('The input of the deprocess_img must be a image with dims of'
                                '[1,height,width,channel] or [height,width,channel]')
    if len(img.shape) != 3:
        raise ValueError('Invalid Input to deprocess_img')
    
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    
    img = np.clip(img,0,255).astype('uint8')
    return img

def saveimg(bestimg,path):
    img = Image.fromarray(bestimg)
    img.save(path)


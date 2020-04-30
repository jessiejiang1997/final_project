#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import image
from datetime import datetime
from tensorflow import keras
import color_transfer
                  
content_layers = ['block4_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def model_init():
    """
    this function will load pretrained(imagenet) vgg19 model and give access to output of intermedia layer
    then it will initialize a new model which take a picture as a input and output a list of vgg19 layer output.
    
    Return:
    return a model that input a picture and output the content feature and style feature
    """
    
    vgg19 = tf.keras.applications.VGG19(include_top = False,weights = 'imagenet')
    vgg19.trainable = False
    content_output = []
    style_output = []

    for layer in content_layers:
        content_output.append(vgg19.get_layer(layer).output)
    
    for layer in style_layers:
        style_output.append(vgg19.get_layer(layer).output)

    output = content_output + style_output
    model = keras.models.Model(vgg19.input,output)
    
    return model    


def content_loss(base_content,target):
    c_loss = tf.reduce_mean(tf.square(base_content - target))/2
    return c_loss

def gram_matrix(input_tensor):
    channel = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor,[-1,channel]) 
    gram = tf.matmul(a,a,transpose_a = True)
    return gram

def style_loss(base_style,target):
    a = gram_matrix(base_style)
    b = gram_matrix(target)
    h,w,c = base_style.get_shape().as_list()
    s_loss = tf.reduce_mean(tf.square(a - b))/(4*(h**2)*(w**2)*(c**2))
    return s_loss


def get_feature(model, style_paths, content_path):

    styles = []
    for style_path in style_paths:
        style = image.pre_process_img(style_path)
        style = np.squeeze(style, axis=0)
        styles.append(style)

    content = image.pre_process_img(content_path)
    # creating new style matrix for color transfer
    content = np.squeeze(content, axis=0)
    # print("Line 70: style's shape", style.shape)
    new_styles = []
    for style in styles:
        new_style = color_transfer.pixel_transformation('image_analogies', style, content)
        new_style = np.expand_dims(new_style, axis=0)
        new_styles.append(new_style)
    # print("Line 74: new_style's shape", new_style.shape)
    content = np.expand_dims(content, axis=0)

    artist_style_feature_outputs = []
    for new_style in new_styles:
        style_feature_outputs = model(new_style)
        artist_style_feature_outputs.append(style_feature_outputs)
    content_feature_outputs = model(content)
    
    style_feature_arr, content_feature_arr = [], []

    for image_style_feature_outputs in artist_style_feature_outputs:
        for feature in image_style_feature_outputs[num_content_layers:]:
            style_feature_arr.append(feature[0])
    
    for feature in content_feature_outputs[:num_content_layers]:
        content_feature_arr.append(feature[0])
    
    # img = image.pre_process_img(path)
    # feature_outputs = model(img)
    # feature_arr = []

    # if mode == 'style':
    #     for feature in feature_outputs[num_content_layers:]:
    #         feature_arr.append(feature[0])
       
    # if mode =='content':
    #     for feature in feature_outputs[:num_content_layers]:
    #         feature_arr.append(feature[0])

    return (style_feature_arr, content_feature_arr)
    

def loss(model,loss_weights,init_image,content_features,style_features):

    style_weight,content_weight = loss_weights
    
    # feed the init image in the model,then we would get the 
    # content feature and the style feature from the layers 
    # we desire
    
    features = model(init_image)
    gen_style_feature = features[num_content_layers:]
    gen_content_feature = features[:num_content_layers]
    
    total_style_loss = 0
    total_content_loss = 0
    
    style_layer_weight = 1.0/ float(num_style_layers)
    for i in range(len(style_features)): 
        total_style_loss = total_style_loss + style_layer_weight * style_loss(style_features[i], gen_style_feature[i])
    

    content_layer_weight = 1.0/ float(num_content_layers)
    for i in range(len(content_features)): 
        total_content_loss = total_content_loss + content_layer_weight * content_loss(content_features[i], gen_content_feature[i])
    
    total_style_loss *= style_weight
    total_content_loss *= content_weight
    total_loss = total_style_loss + total_content_loss
    return total_loss,total_content_loss,total_style_loss
    

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        allloss = loss(**cfg)
    #Compute Gradient with respect to the generated image
    total_loss = allloss[0]
    return tape.gradient(total_loss,cfg['init_image']),allloss


def run(content_path,style_path,iteration):

    content_weight = 1e3
    style_weight = 1

    model = model_init()
    for layer in model.layers:
        layer.trainable = False
    
    style_features, content_features = get_feature(model, style_path, content_path)
    print(np.array(style_features).shape)
    print(np.array(content_features).shape)
    
    init_image = image.pre_process_img(content_path) # initialize the generated image with content image
    init_image = tf.Variable(init_image,dtype = tf.float32)
    
    opt = tf.keras.optimizers.Adam(5,beta_1 = 0.99,epsilon = 1e-1)
    
    loss_weights = (content_weight,style_weight)
    
    cfg = {
        'model':model,
        'loss_weights':loss_weights,
        'init_image':init_image,
        'content_features':content_features,
        'style_features':style_features
    }
    #我不太知道这个norm means是怎么来的，image.py里用的也是相同的值，norm means是用来normalize图片的，我看几个github版本用的数值都差不多，但不知道怎么算的
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    #store the loss and the img
    best_loss, best_img = float('inf'), None
    imgs = []
    start = datetime.now()

    for i in range(iteration):
        
        grads, all_loss = compute_grads(cfg)
        losss, content_losss, style_losss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        
        # 以下用了一些image.py里的function，用的github上的代码，之后改掉
        if losss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = losss
            best_img = image.deprocess_img(init_image.numpy())

        if i % 100 == 0:
            end = datetime.now()
            print('[INFO]Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}'
                  .format(losss, style_losss, content_losss))
            print(f'100 iters takes {end -start}')
            start = datetime.now()

            img = init_image.numpy()
            img = image.deprocess_img(img)
            path = 'output_' + str(i) + '.jpg'
            image.saveimg(img, path)
            imgs.append(img)


    return best_img, best_loss
    
    


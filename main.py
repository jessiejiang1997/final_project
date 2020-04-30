#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import image
import model
import ssl

content_path = 'james.jpg'
style_paths = ['Vincent_van_Gogh_69.jpg', 'Vincent_van_Gogh_604.jpg']

#image.py用的github上的原码，我过几天再更新一版

if __name__ == "__main__":
    best, best_loss = model.run(content_path, style_paths, iteration=200)
    image.saveimg(best, 'best.jpg')

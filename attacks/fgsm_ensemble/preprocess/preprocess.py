"""Preprocess data
referenced https://github.com/rwightman/tensorflow-litterbox/blob/master/litterbox/fabric/image_processing_common.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

IMAGENET_MEAN_CAFFE = [103.939, 116.779, 123.68]
IMAGENET_MEAN_STD = [
    [0.485, 0.456, 0.406],  # mean
    [0.229, 0.224, 0.225],  # std
]


def image_normalize(
        image,
        method='global',
        global_mean_std=IMAGENET_MEAN_STD,
        caffe_mean=IMAGENET_MEAN_CAFFE):
    """
    Args:
        image:
        method:
        global_mean_std:
        caffe_mean:
    Returns:
    """
    if method == 'caffe' or method == 'caffe_bgr':
        print('Caffe BGR normalize', image.get_shape())
        # Rescale to [0, 255]
        image = tf.multiply(image, 255.0)
        # Convert RGB to BGR
        red, green, blue = tf.split(image, [1,1,1], 3)
        image = tf.concat([blue, green, red], 3)
        image = tf.subtract(image, caffe_mean)
    elif method == 'caffe_rgb':
        print('Caffe RGB normalize', image.get_shape())
        # Rescale to [0, 255]
        image = tf.multiply(image, 255.0)
        caffe_mean_rgb = tf.gather(caffe_mean, [2, 1, 0])
        image = tf.subtract(image, caffe_mean_rgb)
    elif method == 'frame':
        print("Per-frame standardize", image.get_shape())
        mean, var = tf.nn.moments(image, axes=[0, 1], shift=0.3)
        std = tf.sqrt(tf.add(var, .001))
        image = tf.subtract(image, mean)
        image = tf.div(image, std)
    elif method == 'global':
        print('Global standardize', image.get_shape())
        image = tf.subtract(image, global_mean_std[0])
        image = tf.div(image, global_mean_std[1])
    else:
        assert method == 'default'
        print('Default normalize [-1, 1]', image.get_shape())
        # Rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
    return image


def image_invert(
        image,
        method='global',
        global_mean_std=IMAGENET_MEAN_STD,
        caffe_mean=IMAGENET_MEAN_CAFFE):
    """
    Args:
        image:
        method:
        global_mean_std:
        caffe_mean:
    Returns:
    """
    if method == 'caffe' or method == 'caffe_bgr':
        print('Caffe BGR invert', image.get_shape())
        image = tf.add(image, caffe_mean)
        blue, green, red = tf.split(image, [1,1,1], 3)
        image = tf.concat([red, green, blue], 3)
    elif method == 'caffe_rgb':
        print('Caffe RGB invert', image.get_shape())
        caffe_mean_rgb = tf.gather(caffe_mean, [2, 1, 0])
        image = tf.add(image, caffe_mean_rgb)
    elif method == 'global':
        print('Global standardize invert', image.get_shape())
        image = tf.multiply(image, global_mean_std[1])
        image = tf.add(image, global_mean_std[0])
        image = tf.multiply(image, 255)
    else:
        assert method == 'default'
        print('Default invert [-1, 1] ', image.get_shape())
        # Rescale to [-1,1] instead of [0, 1)
        image = tf.multiply(image, 0.5)
        image = tf.add(image, 0.5)
        image = tf.multiply(image, 255.0)
    image = tf.clip_by_value(image, 0, 255)
    return image


"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from nets import inception
from nets import resnet_v1
from nets import vgg
from preprocess.preprocess import image_normalize
import numpy as np
slim = tf.contrib.slim


normalization_method = ['default','default','default','default','global',
                        'caffe_rgb','caffe_rgb','default','default','caffe_rgb',
                        'caffe_rgb']

class InceptionV1(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[0])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(inception.inception_v1_arg_scope()):
      _, end_points = inception.inception_v1(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class InceptionV2(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[1])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(inception.inception_v2_arg_scope()):
      _, end_points = inception.inception_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class InceptionV3(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[2])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class InceptionV4(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[3])
    with slim.arg_scope(inception.inception_v4_arg_scope()):
      _, end_points = inception.inception_v4(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return output


class InceptionResnetV2(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[4])
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
      _, end_points = inception.inception_resnet_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return output


class ResnetV1_101(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[5])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_101(
          x_input, num_classes=self.num_classes-1, is_training=False,
          reuse=reuse)
    self.built = True
    end_points['predictions'] = \
                  tf.concat([tf.zeros([tf.shape(x_input)[0], 1]), 
                                  tf.reshape(end_points['predictions'], [-1, 1000])], 
                                  axis=1)
    output = end_points['predictions']
    # Strip off the extra reshape op at the output
    return output


class ResnetV1_152(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[6])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_152(
          x_input, num_classes=self.num_classes-1, is_training=False,
          reuse=reuse)
    self.built = True
    end_points['predictions'] = \
                  tf.concat([tf.zeros([tf.shape(x_input)[0], 1]), 
                                  tf.reshape(end_points['predictions'], [-1, 1000])], 
                                  axis=1)
    output = end_points['predictions']
    # Strip off the extra reshape op at the output
    return output


class ResnetV2_101(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[7])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      _, end_points = resnet_v2.resnet_v2_101(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class ResnetV2_152(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[8])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      _, end_points = resnet_v2.resnet_v2_152(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class Vgg_16(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[9])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(vgg.vgg_arg_scope()):
      _, end_points = vgg.vgg_16(
          x_input, num_classes=1000, is_training=False)
    self.built = True
    end_points['predictions'] = tf.nn.softmax(end_points['vgg_16/fc8'])
    end_points['predictions'] = \
                  tf.concat([tf.zeros([tf.shape(x_input)[0], 1]), 
                                  tf.reshape(end_points['predictions'], [-1, 1000])], 
                                  axis=1)
    output = end_points['predictions']
    return output


class Vgg_19(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    x_input = image_normalize(x_input, normalization_method[10])
    x_input = tf.image.resize_images(x_input, [224, 224])
    with slim.arg_scope(vgg.vgg_arg_scope()):
      _, end_points = vgg.vgg_19(
          x_input, num_classes=1000, is_training=False)
    end_points['predictions'] = tf.nn.softmax(end_points['vgg_19/fc8'])
    end_points['predictions'] = \
                  tf.concat([tf.zeros([tf.shape(x_input)[0], 1]), 
                                  tf.reshape(end_points['predictions'], [-1, 1000])], 
                                  axis=1)
    self.built = True
    output = end_points['predictions']
    return output


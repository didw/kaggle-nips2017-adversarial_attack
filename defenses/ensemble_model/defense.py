"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from nets import inception
from nets import resnet_v1
from nets import vgg
from scipy.ndimage import zoom
from preprocess.preprocess import image_normalize
from model_list import InceptionV1, InceptionV2, InceptionV3, InceptionV4, InceptionResnetV2, ResnetV1_101, ResnetV1_152, ResnetV2_101, ResnetV2_152, Vgg_16, Vgg_19

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v1', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v1_101', '', 'Path to checkpoint for resnet_v2_101 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v1_152', '', 'Path to checkpoint for resnet_v2_101 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_101', '', 'Path to checkpoint for resnet_v2_101 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_152', '', 'Path to checkpoint for resnet_v2_152 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg_16', '', 'Path to checkpoint for resnet_v2_152 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg_19', '', 'Path to checkpoint for resnet_v2_152 network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 100, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'test_idx', 0, 'Which version to test. 0 for all')

tf.flags.DEFINE_string(
    'ensemble_type', 'mean', 'Ensemble type (mean, vote)')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    #images[idx, :, :, :] = image * 2.0 - 1.0
    images[idx, :, :, :] = image
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001
  ensemble_type = FLAGS.ensemble_type

  tf.logging.set_verbosity(tf.logging.INFO)

  checkpoint_path_list = [FLAGS.checkpoint_path_inception_v1,
                          FLAGS.checkpoint_path_inception_v2,
                          FLAGS.checkpoint_path_inception_v3,
                          FLAGS.checkpoint_path_inception_v4,
                          FLAGS.checkpoint_path_inception_resnet_v2,
                          FLAGS.checkpoint_path_resnet_v1_101,
                          FLAGS.checkpoint_path_resnet_v1_152,
                          FLAGS.checkpoint_path_resnet_v2_101,
                          FLAGS.checkpoint_path_resnet_v2_152,
                          FLAGS.checkpoint_path_vgg_16,
                          FLAGS.checkpoint_path_vgg_19]
  normalization_method = ['default','default','default','default','global',
                          'caffe_rgb','caffe_rgb','default','default','caffe_rgb',
                          'caffe_rgb']


  print("Build Graph..")
  graph_list = []
  sess_list = []
  x_input_list = []
  prob_list = []
  for i in range(len(checkpoint_path_list)):
    graph = tf.Graph()
    graph_list.append(graph)
    with graph.as_default():
      x_input_list.append(tf.placeholder(tf.float32, shape=batch_shape))
      prob_list.append(None)
      if i in [0,1,2,5,6,9]: continue
      if i == 0:
        model = InceptionV1(num_classes)
      if i == 1:
        model = InceptionV2(num_classes)
      if i == 2:
        model = InceptionV3(num_classes)
      if i == 3:
        model = InceptionV4(num_classes)
      if i == 4:
        model = InceptionResnetV2(num_classes)
      if i == 5:
        model = ResnetV1_101(num_classes)
      if i == 6:
        model = ResnetV1_152(num_classes)
      if i == 7:
        model = ResnetV2_101(num_classes)
      if i == 8:
        model = ResnetV2_152(num_classes)
      if i == 9:
        model = Vgg_16(num_classes)
      if i == 10:
        model = Vgg_19(num_classes)
      prob_list[i] = model(x_input_list[i])


  print("Build Session..")
  for i in range(len(checkpoint_path_list)):
    sess_list.append(None)
    if i in [0,1,2,5,6,9]: continue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = graph_list[i]
    sess_list[i] = tf.Session(graph=graph, config=config)

  print("Loading model..")
  for i in range(len(checkpoint_path_list)):
    if i in [0,1,2,5,6,9]: continue
    graph = graph_list[i]
    sess = sess_list[i]
    with sess.as_default():
      with graph.as_default():
        model_saver = tf.train.Saver(tf.global_variables())
        model_saver.restore(sess, checkpoint_path_list[i])

  print("Run..")
  label_list = []
  filenames_list = []
  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    if len(label_list) == len(filenames_list):
      filenames_list.extend(filenames)
    pred_list = []
    for i in xrange(len(checkpoint_path_list)):
      if i in [0,1,2,5,6,9]: continue
      sess = sess_list[i]
      with sess.as_default():
        pred = sess.run(prob_list[i], feed_dict={x_input_list[i]: images})
        pred_list.append(pred)
    #print("np.shape(pred_list):", np.shape(pred_list))
    pred = np.mean(pred_list, axis=0)  # model x batch x class
    #print("np.shape(pred):", np.shape(pred))
    label_list.extend(np.argmax(pred, axis=1)) # model_num X batch X class_num ==(np.mean)==> batch X class_num ==(np.argmax)==> batch
    #print("np.shape(label_list):", np.shape(label_list))

  with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
    for filename, label in zip(filenames_list, label_list):
      out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()

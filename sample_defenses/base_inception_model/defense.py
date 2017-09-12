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
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_101', '', 'Path to checkpoint for resnet_v2_101 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_152', '', 'Path to checkpoint for resnet_v2_152 network.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg', '', 'Path to checkpoint for vgg network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

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
    images[idx, :, :, :] = image * 2.0 - 1.0
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

  tf.logging.set_verbosity(tf.logging.INFO)

  graph_inception = tf.Graph()
  graph_resnet_v2_101 = tf.Graph()
  graph_resnet_v2_152 = tf.Graph()
  graph_vgg19 = tf.Graph()

  checkpoint_path_list = [FLAGS.checkpoint_path_inception,
                          FLAGS.checkpoint_path_resnet_v2_101,
                          FLAGS.checkpoint_path_resnet_v2_152,
                          FLAGS.checkpoint_path_vgg]
  pred_list = []
  for idx, checkpoint_path in zip([1,2,3,4], checkpoint_path_list):
    if idx != 1:
      continue
    with tf.Graph().as_default():
      # Prepare graph
      x_input = tf.placeholder(tf.float32, shape=batch_shape)

      if idx == 1:
        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 2:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          _, end_points = resnet_v2.resnet_v2_101(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 3:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          _, end_points = resnet_v2.resnet_v2_152(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 4:
        with slim.arg_scope(vgg.vgg_arg_scope()):
          _, end_points = vgg.vgg_16(
              x_input, num_classes=num_classes, is_training=False,
              dropout_keep_prob=1.0, spatial_squeeze=True)

      #end_points = tf.reduce_mean([end_points1['Predictions'], end_points2['Predictions'], end_points3['Predictions'], end_points4['Predictions']], axis=0)

      #predicted_labels = tf.argmax(end_points, 1)

      # Run computation
      saver = tf.train.Saver(slim.get_model_variables())
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=tf.train.Scaffold(saver=saver),
          checkpoint_filename_with_path=checkpoint_path,
          master=FLAGS.master)

      pred_in = []
      filenames_list = []
      with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
          for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            filenames_list.extend(filenames)
            end_points_dict = sess.run(end_points, feed_dict={x_input: images})
            try:
              pred_in.extend(end_points_dict['Predictions'].reshape(-1, num_classes))
            except KeyError:
              pred_in.extend(end_points_dict['predictions'].reshape(-1, num_classes))
            # pred_list = model num x batch x class num
      pred_list.append(pred_in)

    print(np.shape(pred_list))
    pred = np.mean(pred_list, axis=0)
    print(np.shape(pred))
    labels = np.argmax(pred, axis=1) # model_num X batch X class_num ==(np.mean)==> batch X class_num ==(np.argmax)==> batch
    print(np.shape(labels))
    for filename, label in zip(filenames_list, labels):
      out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()

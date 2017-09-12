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
  pred_list = []
  for idx, checkpoint_path in enumerate(checkpoint_path_list, 1):
    with tf.Graph().as_default():
      if idx in [1,2,3,4,5,6,7]:
        continue
      # Prepare graph
      if idx in [1,2,6,7,10,11]:
        _x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_input = tf.image.resize_images(_x_input, [224, 224])
      else:
        _x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_input = _x_input

      if idx == 1:
        with slim.arg_scope(inception.inception_v1_arg_scope()):
          _, end_points = inception.inception_v1(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 2:
        with slim.arg_scope(inception.inception_v2_arg_scope()):
          _, end_points = inception.inception_v2(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 3:
        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 4:
        with slim.arg_scope(inception.inception_v4_arg_scope()):
          _, end_points = inception.inception_v4(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 5:
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
          _, end_points = inception.inception_resnet_v2(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 6:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
          _, end_points = resnet_v1.resnet_v1_101(
              x_input, num_classes=1000, is_training=False)
      elif idx == 7:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
          _, end_points = resnet_v1.resnet_v1_152(
              x_input, num_classes=1000, is_training=False)
      elif idx == 8:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          _, end_points = resnet_v2.resnet_v2_101(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 9:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          _, end_points = resnet_v2.resnet_v2_152(
              x_input, num_classes=num_classes, is_training=False)
      elif idx == 10:
        with slim.arg_scope(vgg.vgg_arg_scope()):
          _, end_points = vgg.vgg_16(
              x_input, num_classes=1000, is_training=False)
          end_points['predictions'] = tf.nn.softmax(end_points['vgg_16/fc8'])
      elif idx == 11:
        with slim.arg_scope(vgg.vgg_arg_scope()):
          _, end_points = vgg.vgg_19(
              x_input, num_classes=1000, is_training=False)
          end_points['predictions'] = tf.softmax(end_points['vgg_19/fc8'])

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
            #if idx in [1,2,6,7,10,11]:
            #  # 16x299x299x3
            #  images = zoom(images, (1, 0.7491638795986622, 0.7491638795986622, 1), order=2)
            filenames_list.extend(filenames)
            end_points_dict = sess.run(end_points, feed_dict={_x_input: images})
            if idx in [6,7,10,11]:
              end_points_dict['predictions'] = \
                            np.concatenate([np.zeros([FLAGS.batch_size, 1]), 
                                            np.array(end_points_dict['predictions'].reshape(-1, 1000))], 
                                            axis=1)
            try:
              pred_in.extend(end_points_dict['Predictions'].reshape(-1, num_classes))
            except KeyError:
              pred_in.extend(end_points_dict['predictions'].reshape(-1, num_classes))
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

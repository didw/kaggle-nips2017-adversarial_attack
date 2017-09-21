"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from nets import inception
from nets import resnet_v1
from nets import vgg
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
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

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
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def load_total_labels(fname):
  df = pd.read_csv(fname)
  return df


def load_labels(filenames, label):
  fid = map(lambda x: x.replace('.png', ''), filenames)
  fid = pd.DataFrame(fid, columns=['ImageId'])
  lab = np.array(fid.merge(label)['TrueLabel'])
  one_hot = np.zeros((len(filenames), 1001))
  one_hot[np.arange(len(filenames)), lab] = 1
  return one_hot


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  total_labels = load_total_labels('images.csv')

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

  tf.logging.set_verbosity(tf.logging.INFO)

  graph_list = []
  sess_list = []
  x_input_list = []
  y_list = []
  prob_list = []
  for i in range(len(checkpoint_path_list)):
    graph = tf.Graph()
    with graph.as_default():
      x_input_list.append(tf.placeholder(tf.float32, shape=batch_shape))
      y_list.append(tf.placeholder(tf.float32, shape=[FLAGS.batch_size, num_classes]))
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
      prob_list.append(model(x_input_list[i]))
    graph_list.append(graph)


  for i in range(len(checkpoint_path_list)):
    graph = graph_list[i]
    sess_list.append(tf.Session(graph=graph))

  for i in range(len(checkpoint_path_list)):
    graph = graph_list[i]
    sess = sess_list[i]
    with sess.as_default():
      with graph.as_default():
        model_saver = tf.train.Saver(tf.global_variables())
        model_saver.restore(sess, checkpoint_path_list[i])


  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    #adv_images = make_noise(images, FLAGS.max_epsilon)
    for i in xrange(len(checkpoint_path_list)):
      graph = graph_list[i]
      sess = sess_list[i]
      with sess.as_default():
        y_labels = load_labels(filenames, total_labels)
        probs = sess.run(prob_list[i], feed_dict={x_input_list[i]: images})
        print(np.argmax(probs, axis=1))



if __name__ == '__main__':
  tf.app.run()

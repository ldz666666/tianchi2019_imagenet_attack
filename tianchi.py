"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.stats as st
from timeit import default_timer as timer

import tensorflow as tf
from nets import inception_v3, inception_resnet_v2, resnet_v2, resnext

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', '', 'Path to checkpoint for inception network.')
    
tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', '', 'Path to checkpoint for inception network.')
    
tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_101', '', 'Path to checkpoint for inception network.')
    
tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2_50', '', 'Path to checkpoint for inception network.')
    
tf.flags.DEFINE_string(
    'checkpoint_path_resnext_101', '', 'Path to checkpoint for inception network.')

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
    'batch_size', 1, 'How many images process at one time.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')
tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'sig', 4, 'gradient smoothing')
tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_integer(
  'iterations', 13, 'iterations')

FLAGS = tf.flags.FLAGS

def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'dev.csv')) as f:
    target_class = {row[0]: int(row[2]) for row in csv.reader(f) if len(row) >= 2}
  with tf.gfile.Open(os.path.join(input_dir, 'dev.csv')) as f:
    true_label = {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}
  return target_class, true_label


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
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

def input_diversity(input_tensor):
  """
  kernel_size=10
  p_dropout=0.1
  kernel = tf.divide(tf.ones((kernel_size,kernel_size,3,3),tf.float32),tf.cast(kernel_size**2,tf.float32))
  input_shape = input_tensor.get_shape()
  rand = tf.where(tf.random_uniform(input_shape) < tf.constant(p_dropout, shape=input_shape), 
    tf.constant(1., shape=input_shape), tf.constant(0., shape=input_shape))
  image_d = tf.multiply(input_tensor,rand)
  image_s = tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],'SAME')
  input_tensor = tf.add(image_d,tf.multiply(image_s,tf.subtract(tf.cast(1,tf.float32),rand)))
  """
  rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = FLAGS.image_resize - rnd
  w_rem = FLAGS.image_resize - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
  ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
  ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width], 
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return ret

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel

def preprocess_for_model(x):
  red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
  bgr = tf.concat(axis=3, values=[blue, green, red])
  x_resize = tf.image.resize_bilinear(bgr, [224,224],align_corners=False)
  images = tf.transpose(x_resize, [0, 3, 1, 2])
  return images
    
def graph(x, target_class_input, true_label_input, i, x_max, x_min, grad):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = eps / FLAGS.iterations
  momentum = FLAGS.momentum
  num_classes = 1001
  
  x_div = input_diversity(x)

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x_div, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_adv_res_v2, end_points_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x_div, num_classes=num_classes, is_training=False, scope='AdvInceptionResnetV2')
        
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False)
        
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')
  
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_res_v2, end_points_res_v2 = resnet_v2.resnet_v2_50(
        x_div, num_classes=num_classes, is_training=False, scope='resnet_v2_50')
  
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_res_v2_101, end_points_res_v2_101 = resnet_v2.resnet_v2_101(
        x_div, num_classes=num_classes, is_training=False, scope='resnet_v2_101')

  with TowerContext(tower_name='eval', is_training=False):
        logits_advresnext_101 = resnext.ResNeXtDenoiseAllModel().get_logits(preprocess_for_model(x_div))    
            
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)
  one_hot_true_label = tf.one_hot(true_label_input, num_classes)
  one_hot_target_class_resnext = tf.one_hot(target_class_input-1, num_classes-1)
  one_hot_true_label_resnext = tf.one_hot(true_label_input-1, num_classes-1)

    
  logits = (logits_inc_v3 + logits_ens3_adv_v3 + 
    logits_ens4_adv_v3 + logits_ensadv_res_v2 + logits_adv_res_v2 +logits_adv_v3+logits_res_v2+logits_res_v2_101) / 8

  auxlogits = (end_points_inc_v3['AuxLogits'] + end_points_ens3_adv_v3['AuxLogits'] + 
    end_points_ens4_adv_v3['AuxLogits'] + end_points_adv_res_v2['AuxLogits'] + 
    end_points_ensadv_res_v2['AuxLogits'] +end_points_adv_v3['AuxLogits']+logits_res_v2+logits_res_v2_101) / 8
  
  logits_resnext = logits_advresnext_101
    
  cross_entropy = -tf.losses.softmax_cross_entropy(one_hot_true_label,
                                                     logits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
  cross_entropy -= tf.losses.softmax_cross_entropy(one_hot_true_label,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
  cross_entropy2 = tf.losses.softmax_cross_entropy(one_hot_true_label_resnext,
                                                  logits_resnext,
                                                  label_smoothing=0.0,
                                                  weights=1.0)

  '''cross_entropy -= tf.losses.softmax_cross_entropy(one_hot_true_label,
                                                     logits_ensadv_res_v2,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
  cross_entropy -= tf.losses.softmax_cross_entropy(one_hot_true_label,
                                                     end_points_ensadv_res_v2['AuxLogits'],
                                                     label_smoothing=0.0,
                                                     weights=0.4)'''
  rnd = i%2
  cross_entropy = tf.cond(tf.greater(rnd,0),lambda:cross_entropy,lambda:cross_entropy2)
  noise = tf.gradients(cross_entropy, x)[0]

  kernel = gkern(7, FLAGS.sig).astype(np.float32)
  #kernel = gkern(15, 4).astype(np.float32)
  stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
  stack_kernel = np.expand_dims(stack_kernel, 3)
  
  noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), 
    [FLAGS.batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), 
    [FLAGS.batch_size, 1, 1, 1])
  x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, target_class_input, true_label_input, i, x_max, x_min, noise


def stop(x, target_class_input, true_label_input, i, x_max, x_min, grad):
  return tf.less(i, FLAGS.iterations)

#by heziwen
'''def all_16(x, y):
  x = tf.sign(x-y)*32/255.0 + y
  x = tf.clip_by_value(x, -1.0, 1.0)
  return x'''
    
def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  full_start = timer()
  eps = 2.0 * FLAGS.max_epsilon / 255.0

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class, all_images_true_label = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    true_label_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)

    x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, target_class_input,true_label_input, i, x_max, x_min, grad])
    s1 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionResnetV2'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
    s5 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
    s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))
    #s4 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
    model = resnext.ResNeXtDenoiseAllModel()
    print('Created Graph')
    
    #byheziwen
    '''x_input1 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input2 = tf.placeholder(tf.float32, shape=batch_shape)
    x_all_16 = all_16(x_input1, x_input2)'''
    
    # Run computation
    with tf.Session() as sess:
      processed = 0.0
      s1.restore(sess, FLAGS.checkpoint_path_adv_inception_resnet_v2)
      s2.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
      s3.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
      s4.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
      s5.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
      s6.restore(sess, FLAGS.checkpoint_path_inception_v3)
      s7.restore(sess, FLAGS.checkpoint_path_resnet_v2_101)
      s8.restore(sess, FLAGS.checkpoint_path_resnet_v2_50)     
      #s4.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
      model = get_model_loader(FLAGS.checkpoint_path_resnext_101).init(sess)
      print('Initialized Models')

      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_taget_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))
        true_label_for_batch = (
            [all_images_true_label[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))
        adv_images = sess.run(x_adv, feed_dict={x_input: images, target_class_input: target_class_for_batch, true_label_input:true_label_for_batch})
        #by heziwen
        #adv_images = sess.run(x_all_16, feed_dict={x_input1:adv_images, x_input2:images})
        
        save_images(adv_images, filenames, FLAGS.output_dir)
        processed += FLAGS.batch_size
      full_end = timer()
      print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))

if __name__ == '__main__':
  tf.app.run()

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf

from tensorpack.models import regularize_cost, BatchNorm
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import get_current_tower_context, TowerFuncWrapper
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once
from tensorpack.tfutils.collection import freeze_collection
from tensorpack.tfutils.varreplace import custom_getter_scope

from third_party.imagenet_utils import ImageNetModel


IMAGE_SCALE = 2.0 / 255


class NoOpAttacker():
    """
    A placeholder attacker which does nothing.
    """
    def attack(self, image, label, model_func):
        return image, -tf.ones_like(label)





class AdvImageNetModel(ImageNetModel):

    """
    Feature Denoising, Sec 5:
    A label smoothing of 0.1 is used.
    """
    label_smoothing = 0.1

    def set_attacker(self, attacker):
        self.attacker = attacker

    def build_graph(self, image, label):
        """
        The default tower function.
        """
        image = self.image_preprocess(image)
        assert self.data_format == 'NCHW'
        image = tf.transpose(image, [0, 3, 1, 2])
        ctx = get_current_tower_context()

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # BatchNorm always comes with trouble. We use the testing mode of it during attack.
            with freeze_collection([tf.GraphKeys.UPDATE_OPS]), argscope(BatchNorm, training=False):
                image, target_label = self.attacker.attack(image, label, self.get_logits)
                image = tf.stop_gradient(image, name='adv_training_sample')

            logits = self.get_logits(image)

        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)
        AdvImageNetModel.compute_attack_success(logits, target_label)
        if not ctx.is_training:
            return

        wd_loss = regularize_cost(self.weight_decay_pattern,
                                  tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        total_cost = tf.add_n([loss, wd_loss], name='cost')

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    def get_inference_func(self, attacker):
        """
        Returns a tower function to be used for inference. It generates adv
        images with the given attacker and runs classification on it.
        """

        def tower_func(image, label):
            assert not get_current_tower_context().is_training
            image = self.image_preprocess(image)
            image = tf.transpose(image, [0, 3, 1, 2])
            image, target_label = attacker.attack(image, label, self.get_logits)
            logits = self.get_logits(image)
            ImageNetModel.compute_loss_and_error(logits, label)  # compute top-1 and top-5
            AdvImageNetModel.compute_attack_success(logits, target_label)

        return TowerFuncWrapper(tower_func, self.get_inputs_desc())

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            # For the purpose of adversarial training, normalize images to [-1, 1]
            image = image * IMAGE_SCALE - 1.0
            return image

    @staticmethod
    def compute_attack_success(logits, target_label):
        """
        Compute the attack success rate.
        """
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        equal_target = tf.equal(pred, target_label)
        success = tf.cast(equal_target, tf.float32, name='attack_success')
        add_moving_summary(tf.reduce_mean(success, name='attack_success_rate'))

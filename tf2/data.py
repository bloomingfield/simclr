# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from pdb import set_trace as pb
import numpy as np

FLAGS = flags.FLAGS


def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      # pb()
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # dataset = builder.as_dataset(
    #     split=FLAGS.train_split if is_training else FLAGS.eval_split,
    #     shuffle_files=is_training,
    #     as_supervised=True,
    #     # Passing the input_context to TFDS makes TFDS read different parts
    #     # of the dataset on different workers. We also adjust the interleave
    #     # parameters to achieve better performance.
    #     read_config=tfds.ReadConfig(
    #         interleave_cycle_length=32,
    #         interleave_block_length=1,
    #         input_context=input_context))
    # =============================================================
    if is_training: # is_training
      dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=False, as_supervised=False) # shuffle_files=is_training, as_supervised=True
      
      dataset = dataset.cache()
      dataset_iter =  dataset.batch(2000) #tfds.as_numpy(dataset)
      
      ex_id = np.concatenate([x['id'] for x in dataset_iter])
      ex_id_sorted = np.argsort(ex_id)
      unsort = np.empty_like(ex_id_sorted)
      unsort[ex_id_sorted] = np.arange(ex_id_sorted.size)

      # dataset = dataset.shuffle()
      all_exs_features = np.concatenate([x['image'] for x in dataset_iter])
      all_exs_ind = np.concatenate([x['label'] for x in dataset_iter])
      all_exs_ind_sorted = all_exs_ind[ex_id_sorted]

      if FLAGS.train_mode == 'finetune':
        labels_per_class = 25
        classes = 10

        x = np.array(list(range(0, len(all_exs_ind))))
        balanced_index = []
        rng = np.random.default_rng(0)
        for i in range(classes):
            y_p = all_exs_ind_sorted == i
            y_i = rng.choice(x[y_p], size=labels_per_class, replace=False)
            balanced_index.append(y_i)
        balanced_index = np.concatenate(balanced_index)
        balanced_index = np.sort(balanced_index)
        balanced_index = ex_id_sorted[balanced_index]
        print('===============images chosen===============')
        print(ex_id[balanced_index])
        print('================images chosen================')
        all_exs_ind= np.array([all_exs_ind[i] for i in balanced_index])
        all_exs_features  = np.array([all_exs_features[i] for i in balanced_index])
        dataset = tf.data.Dataset.from_tensor_slices((all_exs_features, all_exs_ind))
      else:
        all_exs_ind= all_exs_ind[ex_id_sorted]
        all_exs_features  = all_exs_features[ex_id_sorted]

        rng = np.random.default_rng(10)
        permute = rng.permutation(ex_id_sorted.shape[0])
        dataset = tf.data.Dataset.from_tensor_slices((all_exs_features[permute], all_exs_ind[permute]))
    else:
      # dataset = builder.as_dataset(
      #   split=FLAGS.train_split if is_training else FLAGS.eval_split,
      #   shuffle_files=is_training, as_supervised=True) # shuffle_files=is_training, as_supervised=True
      dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training, #is_training
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    # =============================================================
    
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      if not FLAGS.deterministic:
        dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) # , num_parallel_calls=tf.data.experimental.AUTOTUNE
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)

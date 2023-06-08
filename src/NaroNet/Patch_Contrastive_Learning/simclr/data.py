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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
import random

import NaroNet.Patch_Contrastive_Learning.simclr.data_util as data_util
import tensorflow.compat.v1 as tf
import itertools
from concurrent import futures
import numpy as np
import random as rand

FLAGS = flags.FLAGS


def pad_to_batch(dataset, batch_size):
  """Pad Tensors to specified batch size.

  Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

  Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

  Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
  """
  def _pad_to_batch(*args):
    """Given Tensors yielded by a Dataset, pads all to the batch size."""
    flat_args = tf.nest.flatten(args)

    for tensor in flat_args:
      if tensor.shape.ndims is None:
        raise ValueError(
            'Unknown number of dimensions for tensor %s.' % tensor.name)
      if tensor.shape.ndims == 0:
        raise ValueError('Tensor %s is a scalar.' % tensor.name)

    # This will throw if flat_args is empty. However, as of this writing,
    # tf.data.Dataset.map will throw first with an internal error, so we do
    # not check this case explicitly.
    first_tensor = flat_args[0]
    first_tensor_shape = tf.shape(first_tensor)
    first_tensor_batch_size = first_tensor_shape[0]
    difference = batch_size - first_tensor_batch_size

    for i, tensor in enumerate(flat_args):
      control_deps = []
      if i != 0:
        # Check that leading dimensions of this tensor matches the first,
        # either statically or dynamically. (If the first dimensions of both
        # tensors are statically known, the we have to check the static
        # shapes at graph construction time or else we will never get to the
        # dynamic assertion.)
        if (first_tensor.shape[:1].is_fully_defined() and
            tensor.shape[:1].is_fully_defined()):
          if first_tensor.shape[0] != tensor.shape[0]:
            raise ValueError(
                'Batch size of dataset tensors does not match. %s '
                'has shape %s, but %s has shape %s' % (
                    first_tensor.name, first_tensor.shape,
                    tensor.name, tensor.shape))
        else:
          curr_shape = tf.shape(tensor)
          control_deps = [tf.Assert(
              tf.equal(curr_shape[0], first_tensor_batch_size),
              ['Batch size of dataset tensors %s and %s do not match. '
               'Shapes are' % (tensor.name, first_tensor.name), curr_shape,
               first_tensor_shape])]

      with tf.control_dependencies(control_deps):
        # Pad to batch_size along leading dimension.
        flat_args[i] = tf.pad(
            tensor, [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1))
      flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

    return tf.nest.pack_sequence_as(args, flat_args)

  return dataset.map(_pad_to_batch)


def build_input_fn_CHURRO_eval(is_training, batch_size, dataset, patch_size):
  """Build input function. 

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True,patch_size=patch_size)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False,patch_size=patch_size)
    
    num_classes=5 # Por poner algo
    # num_classes = builder.info.features['label'].num_classes

    def map_fn(image):
      """Produces multiple transformations of the same batch."""             
      label = tf.one_hot(0, num_classes)
      return image, label, 1.0#, label, 1.0

    def map_fn_file(image, files_names, patches_numbers, marker_mean, patches_position):
      # image, position, marker_mean = dataset.getitem_TEST(np.load(file_name.split('__')[0]),file_name.split('__')[1])
      label = tf.one_hot(0, num_classes)
      index = image.eval(session=tf.Session())   
      return image, patches_position, marker_mean, files_names, patches_numbers, label, 1.0

    # Generate a list of Patches, specifying directory and patch...
    files_names = []
    patches_numbers = []
    patches_position = []
    patches_marker_mean = []
    patches = []
    for n_file in range(len(dataset.files)): # Iterate over images
      image = np.load(dataset.ExperimentFolder+dataset.files[n_file]).squeeze()
      for n_patch in range(dataset.num_patches_inImage[n_file]): # For each Image iterate over patches to get its index.             
        # Extract Patch
        Croppedimage, position, marker_mean = dataset.getitem_TEST(image,n_patch)
        # Concatenate the index to the image
        Croppedimage = np.concatenate((Croppedimage,len(patches)*np.ones((Croppedimage.shape[0],Croppedimage.shape[1],1))),axis=2)
        # Save Patch into dictionary
        files_names.append(dataset.files[n_file]) 
        patches_numbers.append(n_patch)        
        patches_position.append(position)
        patches_marker_mean.append(marker_mean)
        patches.append(Croppedimage)
    
    # Save to dataset
    dataset.save_test_info(files_names,patches_numbers,patches_position,patches_marker_mean)

    # Create dataset using patches
    patches = np.stack(patches)
    patches = np.float32(patches)
    patches = tf.data.Dataset.from_tensor_slices(patches)     
    patches = patches.repeat(-1)    

    # Map dataset to get position, mean marker, etc...
    patches = patches.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    patches = patches.batch(batch_size, drop_remainder=False)
    patches = pad_to_batch(patches,batch_size)
    images, labels, mask = tf.data.make_one_shot_iterator(patches).get_next()
      
    return images, {'labels': labels, 'mask': mask, 'dataset': dataset}    
  return _input_fn

def build_input_fn_CHURRO_eval_nfile(is_training, batch_size, dataset, patch_size,n_file):
  """Build input function. 

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True,patch_size=patch_size)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False,patch_size=patch_size)
    
    num_classes=5 # Por poner algo
    # num_classes = builder.info.features['label'].num_classes

    def map_fn(image):
      """Produces multiple transformations of the same batch."""             
      label = tf.one_hot(0, num_classes)
      return image, label, 1.0#, label, 1.0

    def map_fn_file(image, files_names, patches_numbers, marker_mean, patches_position):
      # image, position, marker_mean = dataset.getitem_TEST(np.load(file_name.split('__')[0]),file_name.split('__')[1])
      label = tf.one_hot(0, num_classes)
      index = image.eval(session=tf.Session())   
      return image, patches_position, marker_mean, files_names, patches_numbers, label, 1.0

    # Generate a list of Patches, specifying directory and patch...
    files_names = []
    patches_numbers = []
    patches_position = []
    patches_marker_mean = []
    patches = []
    # for n_file in range(len(dataset.files)): # Iterate over images
    image = np.load(dataset.path+dataset.files[n_file]).squeeze()
    for n_patch in range(dataset.num_patches_inImage[dataset.files[n_file]]): # For each Image iterate over patches to get its index.             
      # Extract Patch
      Croppedimage, position, marker_mean = dataset.getitem_TEST(image,n_patch)
      # Concatenate the index to the image
      Croppedimage = np.concatenate((Croppedimage,len(patches)*np.ones((Croppedimage.shape[0],Croppedimage.shape[1],1))),axis=2)
      # Save Patch into dictionary
      files_names.append(dataset.files[n_file]) 
      patches_numbers.append(n_patch)        
      patches_position.append(position)
      patches_marker_mean.append(marker_mean)
      patches.append(Croppedimage)
    
    # Save to dataset
    dataset.save_test_info(files_names,patches_numbers,patches_position,patches_marker_mean)

    # Create dataset using patches
    patches = np.stack(patches)
    patches = np.float32(patches)
    patches = tf.data.Dataset.from_tensor_slices(patches)     
    patches = patches.repeat(-1)    

    # Map dataset to get position, mean marker, etc...    
    patches = patches.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    patches = patches.batch(batch_size, drop_remainder=False)
    patches = pad_to_batch(patches,batch_size)
    images, labels, mask = tf.data.make_one_shot_iterator(patches).get_next()
    
    return images, {'labels': labels, 'mask': mask, 'dataset': dataset}
    
  return _input_fn


def load_patches_for_step(is_training, batch_size, dataset, patch_size,n_images_iteration):
  '''
  Build input function. 
  is_training: (boolean) that specifies whether to build in training or eval mode
  batch_size: (int) that specifies the number patches in one epoch
  dataset: (Dataset object) containing functions and info to load images.
  patch_size: (int) the size of the patch
  n_images_iteration: (int) that specifies the number of images to load in each step

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  '''
  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True,patch_size=patch_size)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False,patch_size=patch_size)
    
    num_classes=5 # Por poner algo
    # num_classes = builder.info.features['label'].num_classes

    def map_fn(image):
      """Produces multiple transformations of the same batch."""   
      # Augment Data 
      images = tf.concat([preprocess_fn_pretrain(image),preprocess_fn_pretrain(image)], -1)      
      label = tf.one_hot(0, num_classes)
      return images, label, 1.0#, label, 1.0

    # Prepare to load dataset
    dataset.files = list(dataset.num_patches_inImage.keys())
    indices = list(range(len(dataset.files)))
    rand.shuffle(indices)    
    dataset.files = [dataset.files[r] for r in indices]
    dataset.num_patches_inImage = [dataset.num_patches_inImage[dataset.files[r]] for r in indices]    

    # Get dataset
    data = np.stack([dataset.get_patches_from_image(indx) for indx in range(min(dataset.n_images,n_images_iteration))])    
    data = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    data = np.float32(data)
    data = tf.data.Dataset.from_tensor_slices(data)     
    data = data.repeat(-1)
    data = data.map(map_fn)
    data = data.batch(batch_size, drop_remainder=True)
    data = pad_to_batch(data,batch_size)
    images, labels, mask = tf.data.make_one_shot_iterator(data).get_next()
    
    return images, {'labels': labels, 'mask': mask}
  return _input_fn

def build_input_fn_CHURRO_generator(is_training, batch_size, dataset, patch_size):
  """Build input function. 

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True,patch_size=patch_size)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False,patch_size=patch_size)
    
    num_classes=5 # Por poner algo
    # num_classes = builder.info.features['label'].num_classes

    def map_fn(image):
      """Produces multiple transformations of the same batch."""
      # if FLAGS.train_mode == 'pretrain':     
      print(image)
      image=np.load(dataset.ExperimentFolder+image)
      xs = []
      for _ in range(5): # Number of repetitions.
        im_Cropped=dataset.getItem_Paper(image)        
        xs.append(tf.concat([preprocess_fn_pretrain(im_Cropped),preprocess_fn_pretrain(im_Cropped)], -1))
      images = tf.stack(xs,0)
      

      # label = tf.zeros([num_classes])
      # else:
      #   image = preprocess_fn_finetune(image)
      #   label = tf.one_hot(label, num_classes)
      return images#, label, 1.0

    
    # dataset = builder.as_dataset(
    #     split=FLAGS.train_split if is_training else FLAGS.eval_split,
    #     shuffle_files=is_training, as_supervised=True)
    # if FLAGS.cache_dataset:
    #   dataset = dataset.cache()
    # if is_training:
    #   buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
    #   dataset = dataset.shuffle(batch_size * buffer_multiplier)
    #   dataset = dataset.repeat(-1)
    
    # import multiprocessing as multi
    # from multiprocessing import Manager
    # manager = Manager()
    # glob_data = manager.list([])    
    
    # Obtain Images already augmented. Two patches per Image.
    # pool=futures.ThreadPoolExecutor(12)
    # pool = Pool(5)    
    # images = pool.map(map_fn, list(range(min(batch_size,dataset.n_samples))))
    # images = list(pool.map(map_fn, list(range(min(batch_size,dataset.n_samples)))))    
    data = dataset()
    data = tf.data.TextLineDataset(dataset.files) 
    data = data.repeat(-1)
    data = data.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    images = *map(map_fn, list(range(min(batch_size,dataset.n_samples)))),
    images = tf.concat(images,0)
    nImages = images.shape[0]
    images = tf.data.Dataset.from_tensor_slices(images)    
    images = images.repeat(-1)
    images = images.batch(batch_size, drop_remainder=True)
    images = pad_to_batch(images,batch_size)
    images = tf.concat(images,0)
    # Obtain Iterator.
    images = tf.data.make_initializable_iterator(images).get_next()

    # images = pad_to_batch(images, batch_size)
    
    # Obtain labels of Images
    labels = pad_to_batch(tf.data.Dataset.from_tensor_slices(tf.zeros([int(nImages),num_classes])).repeat(-1).batch(batch_size, drop_remainder=True),batch_size)
    labels = tf.data.make_initializable_iterator(labels).get_next()
    mask = pad_to_batch(tf.data.Dataset.from_tensor_slices(tf.ones([int(nImages)])).repeat(-1).batch(batch_size, drop_remainder=True),batch_size)
    mask = tf.data.make_initializable_iterator(mask).get_next()

    print('Run epoch!!!')
    
    

    # labels = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(min(batch_size,dataset.n_samples))))
    # labels = tf.convert_to_tensor(labels)

    # Obtain labels of Images    
    # labels = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(min(batch_size,dataset.n_samples))))
    # labels = tf.convert_to_tensor(labels)

    # p = multi.Pool(processes=8)
    # globalList = []
    # for i in range(min(batch_size,dataset.n_samples)):
    #   globalList.append(map_fn((i)))
    # p.map(dataset.__,range(min(batch_size,dataset.n_samples)))
    
    return images, {'labels': labels, 'mask': mask}
  return _input_fn

def build_input_fn_CHURRO(is_training, batch_size, dataset, patch_size):
  """Build input function. 

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True,patch_size=patch_size)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False,patch_size=patch_size)
    
    num_classes=5 # Por poner algo
    # num_classes = builder.info.features['label'].num_classes

    def map_fn(image):
      """Produces multiple transformations of the same batch."""
      # if FLAGS.train_mode == 'pretrain':     
      print(image)
      image=np.load(dataset.ExperimentFolder+image)
      xs = []
      for _ in range(5): # Number of repetitions.
        im_Cropped=dataset.getItem_Paper(image)        
        xs.append(tf.concat([preprocess_fn_pretrain(im_Cropped),preprocess_fn_pretrain(im_Cropped)], -1))
      images = tf.stack(xs,0)
      

      # label = tf.zeros([num_classes])
      # else:
      #   image = preprocess_fn_finetune(image)
      #   label = tf.one_hot(label, num_classes)
      return images#, label, 1.0

    
    # dataset = builder.as_dataset(
    #     split=FLAGS.train_split if is_training else FLAGS.eval_split,
    #     shuffle_files=is_training, as_supervised=True)
    # if FLAGS.cache_dataset:
    #   dataset = dataset.cache()
    # if is_training:
    #   buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
    #   dataset = dataset.shuffle(batch_size * buffer_multiplier)
    #   dataset = dataset.repeat(-1)
    
    # import multiprocessing as multi
    # from multiprocessing import Manager
    # manager = Manager()
    # glob_data = manager.list([])    
    
    # Obtain Images already augmented. Two patches per Image.
    # pool=futures.ThreadPoolExecutor(12)
    # pool = Pool(5)    
    # images = pool.map(map_fn, list(range(min(batch_size,dataset.n_samples))))
    # images = list(pool.map(map_fn, list(range(min(batch_size,dataset.n_samples)))))    
    data = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    images = *map(map_fn, list(range(min(batch_size,dataset.n_samples)))),
    images = tf.concat(images,0)
    nImages = images.shape[0]
    images = tf.data.Dataset.from_tensor_slices(images)    
    images = images.repeat(-1)
    images = images.batch(batch_size, drop_remainder=True)
    images = pad_to_batch(images,batch_size)
    images = tf.concat(images,0)
    # Obtain Iterator.
    images = tf.data.make_initializable_iterator(images).get_next()

    # images = pad_to_batch(images, batch_size)
    
    # Obtain labels of Images
    labels = pad_to_batch(tf.data.Dataset.from_tensor_slices(tf.zeros([int(nImages),num_classes])).repeat(-1).batch(batch_size, drop_remainder=True),batch_size)
    labels = tf.data.make_initializable_iterator(labels).get_next()
    mask = pad_to_batch(tf.data.Dataset.from_tensor_slices(tf.ones([int(nImages)])).repeat(-1).batch(batch_size, drop_remainder=True),batch_size)
    mask = tf.data.make_initializable_iterator(mask).get_next()

    print('Run epoch!!!')
    
    

    # labels = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(min(batch_size,dataset.n_samples))))
    # labels = tf.convert_to_tensor(labels)

    # Obtain labels of Images    
    # labels = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(min(batch_size,dataset.n_samples))))
    # labels = tf.convert_to_tensor(labels)

    # p = multi.Pool(processes=8)
    # globalList = []
    # for i in range(min(batch_size,dataset.n_samples)):
    #   globalList.append(map_fn((i)))
    # p.map(dataset.__,range(min(batch_size,dataset.n_samples)))
    
    return images, {'labels': labels, 'mask': mask}
  return _input_fn

def build_input_fn(builder, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
        label = tf.zeros([num_classes])
      else:
        image = preprocess_fn_finetune(image)
        label = tf.one_hot(label, num_classes)
      return image, label, 1.0

    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training, as_supervised=True)
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(map_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params['batch_size'], drop_remainder=is_training)
    dataset = pad_to_batch(dataset, params['batch_size'])
    images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()

    return images, {'labels': labels, 'mask': mask}
  return _input_fn



def get_preprocess_fn(is_training, is_pretrain,patch_size):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  test_crop = False
  return functools.partial(
      data_util.preprocess_image,
      height=patch_size,
      width=patch_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)

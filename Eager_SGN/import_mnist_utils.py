
# coding: utf-8

# ## MNIST Example in [[1609.09106] HyperNetworks](http://blog.otoro.net/2016/09/28/hyper-networks/) ##
# 
# This notebook will reproduce the MNIST experiment in the HyperNetworks paper.  This is a very simple experiment, involving a small modification to the TensorFlow [MNIST tutorial](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html). We will express an MNIST Kernel $W$ with using a vector $z$ consisting of four values, by making $W(z)$ a function of $z$.  This concept can be extended to deeper convnet architectures, such as [Deep Residual Networks](https://github.com/tensorflow/models/tree/master/resnet), if we keep the same $W(z)$ weight generation function for each layer, but learn a different $z$ unique for each layer.  The MNIST experiment in the paper will apply this method to only one layer as a simple proof of concept.
# 
# <img src="https://cdn.rawgit.com/hardmaru/supercell/master/assets/static_hyper_network.svg">
# 
# In addition to HyperNetworks, the process of compressing weights with embedding vectors have been explored before, for example, in these very interesting papers: [Predicting Parameters in Deep Learning](https://arxiv.org/abs/1306.0543) and [Deep Friend Convnets](https://arxiv.org/abs/1412.7149).  In fact, you can even make the weight kernels $W$ a function of the input image, so that the weights can be custom tailored for the input, and this is explored in a recent paper called [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673).  In the HyperNetworks paper, we try to take this a bit further and explore this concept with recurrent networks as well.
# 
# This MNIST example is based off TensorFlow's [MNIST Example](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html) and the architecture is almost identical.
# 
# The only difference is that in a typical convnet architecture, the weights are a trainable tensor:
# 
# `conv_weights = tf.Variable(tf.truncated_normal([f_size, f_size, in_size, out_size], stddev=0.01), name="conv_weights")`
# 
# As you will see later, `conv_weights` will be the output of a hypernetwork instead of a `tf.Variable`.

# In[1]:


# includes

# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import time
import random
from IPython.core.debugger import set_trace
#     set_trace() #this one triggers the debugger
import _pickle as cPickle
import codecs
import collections
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import PIL
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
to_categorical = tf.keras.utils.to_categorical

from tensorflow.python.keras import regularizers
from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import InputSpec


tf.enable_eager_execution()

# misc
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)
weights_folder = 'mnist'
img_size = 28
num_channels = 1
# Support Functions:

# In[2]:


def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)
def orthogonal_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer


# In[3]:


def super_linear(x, output_size, scope=None, reuse=False, init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0):
  # support function doing linear operation.  uses ortho initializer defined earlier.
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()

    w_init = None # uniform
    x_size = shape[1]
    h_size = output_size
    if init_w == "zeros":
      w_init=tf.constant_initializer(0.0)
    elif init_w == "constant":
      w_init=tf.constant_initializer(weight_start)
    elif init_w == "gaussian":
      w_init=tf.random_normal_initializer(stddev=weight_start)
    elif init_w == "ortho":
      w_init=orthogonal_initializer(1.0)

    w = tf.get_variable("super_linear_w",
      [shape[1], output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable("super_linear_b", [output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)


# Useful class to hold MNIST Data and do data augmentation / scrambling / batching.

##% In[4]:


# class to store mnist data
class DataSet(object):
  def __init__(self, images, labels, augment=False):
    # Convert from [0, 255] -> [0.0, 1.0] -> [-1.0, 1.0]
    images = images.astype(np.float32)
    #images = images - 0.5
    #images = 2.0 * images
    self.image_size = 28
    self._num_examples = len(images)
    images = np.reshape(images, (self._num_examples, self.image_size, self.image_size, 1))
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._images = images[perm]
    self._labels = labels[perm]
    self._augment = augment
    self.pointer = 0
    self.upsize = 1 if self._augment else 0
    self.min_upsize = 2
    self.max_upsize = 2
    self.random_perm_mode=False
    self.num_classes = 10

  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples

  def next_batch(self, batch_size=100, with_label = True, one_hot = False):
    if self.pointer >= self.num_examples-2*batch_size:
      self.pointer = 0
    else:
      self.pointer += batch_size
    result = []
    
    upsize_amount = np.random.randint(self.upsize*self.min_upsize, self.upsize*self.max_upsize+1)
    
    #def random_flip(x):
    #  if np.random.rand(1)[0] > 0.5:
    #    return np.fliplr(x)
    #  return x

    def upsize_row_once(img):
      old_size = img.shape[0]
      new_size = old_size+1
      new_img = np.zeros((new_size, img.shape[1], 1))
      rand_row = np.random.randint(1, old_size-1)
      new_img[0:rand_row,:] = img[0:rand_row,:]
      new_img[rand_row+1:,:] = img[rand_row:,:]
      new_img[rand_row,:] = 0.5*(new_img[rand_row-1,:]+new_img[rand_row+1,:])
      return new_img
    def upsize_col_once(img):
      old_size = img.shape[1]
      new_size = old_size+1
      new_img = np.zeros((img.shape[0], new_size, 1))
      rand_col = np.random.randint(1, old_size-1)
      new_img[:,0:rand_col,:] = img[:,0:rand_col,:]
      new_img[:,rand_col+1:,:] = img[:,rand_col:,:]
      new_img[:,rand_col,:] = 0.5*(new_img[:,rand_col-1,:]+new_img[:,rand_col+1,:])
      return new_img
    def upsize_me(img, n=self.max_upsize):
      new_img = img
      for i in range(n):
        new_img = upsize_row_once(new_img)
        new_img = upsize_col_once(new_img)
      return new_img

    for data in self._images[self.pointer:self.pointer+batch_size]:
      result.append(self.distort_image(upsize_me(data, upsize_amount), upsize_amount))
      
    if len(result) != batch_size:
      print ("uh oh, self.pointer = ", self.pointer)
    assert(len(result) == batch_size)
    result_labels = self.labels[self.pointer:self.pointer+batch_size]
    assert(len(result_labels) == batch_size)
    if one_hot:
      result_labels = np.eye(self.num_classes)[result_labels]
    if with_label:
      return self.scramble_batch(np.array(result, dtype=np.float32)), result_labels
    return self.scramble_batch(np.array(result, dtype=np.float32))

  def distort_batch(self, batch, upsize_amount):
    batch_size = len(batch)
    row_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1, batch_size)
    col_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1, batch_size)
    result = np.zeros(shape=(batch_size, self.image_size, self.image_size, 1), dtype=np.float32)
    for i in range(batch_size):
      result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i]+self.image_size, col_distort[i]:col_distort[i]+self.image_size, :]
    return result
  def scramble_batch(self, batch):
    if self.random_perm_mode:
      batch_size = len(batch)
      result = np.copy(batch)
      result = result.reshape(batch_size, self.image_size*self.image_size)
      result = result[:, self.random_key]
      return result
    else:
      result = batch
      return result
  def distort_image(self, img, upsize_amount):
    row_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1)
    col_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1)
    result = np.zeros(shape=(self.image_size, self.image_size, 1), dtype=np.float32)
    result[:, :, :] = img[row_distort:row_distort+self.image_size, col_distort:col_distort+self.image_size, :]
    return result

  def shuffle_data(self):
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._images = self._images[perm]
    self._labels = self._labels[perm]


# In[5]:


# show image of mnist
def show_image(image):
  plt.subplot(1, 1, 1)
  plt.imshow(np.reshape(image, (28, 28)), cmap='Greys', interpolation='nearest')
  plt.axis('off')
  plt.show()


# In[6]:


def show_filter(w_orig):
  w = w_orig.T
  the_shape = w_orig.shape
  print (the_shape)
  f_size = the_shape[0]
  in_dim =the_shape[2]
  out_dim = the_shape[3]
  print ("mean =", np.mean(w))
  print ("stddev =", np.std(w))
  print ("max =", np.max(w))
  print ("min =", np.min(w))
  print ("median =", np.median(w))
  canvas = np.zeros(((f_size+1)*out_dim, (f_size+1)*in_dim))
  for i in range(out_dim):
    for j in range(in_dim):
      canvas[i*(f_size+1):i*(f_size+1)+f_size,j*(f_size+1):j*(f_size+1)+f_size] = w[i, j]
  plt.figure(figsize=(16, 16))
  canvas_fixed = np.zeros((canvas.shape[0]+1,canvas.shape[1]+1))
  canvas_fixed[1:,1:] = canvas
  plt.imshow(canvas_fixed.T, cmap='Greys', interpolation='nearest')
  plt.axis('off')


# In[7]:


def read_data_sets(mnist_data):

  class DataSets(object):
    pass
  data_sets = DataSets()

  data_sets.train = DataSet(mnist_data.train.images, mnist_data.train.labels, augment=True)
  data_sets.valid = DataSet(mnist_data.validation.images, mnist_data.validation.labels, augment=False)
  data_sets.test = DataSet(mnist_data.test.images, mnist_data.test.labels, augment=False)
  XDIM = data_sets.train.image_size
  #random_key = np.random.permutation(XDIM*XDIM)
  #data_sets.train.random_key = random_key
  #data_sets.valid.random_key = random_key
  #data_sets.test.random_key = random_key
  return data_sets


# In[8]:


# # Definte the model
# model = []

# model = Sequential()

# model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
# model.add(Convolution2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
 
# model.add(Flatten())
# model.add(gated_dense(128, 10, activation='relu'))
# # model.add(Dense(128, activation='relu'))

# model.add(Dropout(0.5))
# # model.add(gated_dense(10, 10, activation='softmax'))
# model.add(Dense( 10, activation='softmax'))


# In[9]:


# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#one hot 
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

# In[10]:

def random_batch(x_data, y_data, num_of_training_examples, batch_size=32):
    """
    Create a random batch of training-data.

    :param batch_size: Number of images in the batch.
    :return: 3 numpy arrays (x, y, y_cls)
    """
    num_train = num_of_training_examples
    # Create a random index into the training-set.
    idx = np.random.randint(low=0, high=num_train, size=batch_size)

    # Use the index to lookup random training-data.
    x_batch = x_data[idx]
    y_batch = y_data[idx]
    # y_batch_cls = self.y_train_cls[idx]

    return x_batch, y_batch

def disp_weights(weights):
    plt.imshow(weights)
    cb = plt.colorbar()
    cb.set_label('Weights')
    plt.show()

def plot_example_errors(cls_pred, correct):

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])




"""
Step: 0 accuracy for training: 12.719  and testing 14.031
Step: 1000 accuracy for training: 90.969  and testing 92.312
Step: 2000 accuracy for training: 92.312  and testing 93.094
Step: 3000 accuracy for training: 93.281  and testing 93.344
Step: 0 accuracy for training: 93.219  and testing 93.781
Step: 5000 accuracy for training: 94.406  and testing 94.938
Step: 6000 accuracy for training: 94.750  and testing 95.625
Step: 7000 accuracy for training: 95.375  and testing 95.812
Step: 0 accuracy for training: 97.375  and testing 96.656
Step: 1000 accuracy for training: 97.594  and testing 96.844
Step: 2000 accuracy for training: 97.938  and testing 97.344
Step: 3000 accuracy for training: 97.906  and testing 97.281
Step: 4000 accuracy for training: 98.125  and testing 97.562
Step: 5000 accuracy for training: 97.812  and testing 97.375
Step: 6000 accuracy for training: 98.562  and testing 97.156
Step: 7000 accuracy for training: 98.031  and testing 97.781
Step: 8000 accuracy for training: 98.250  and testing 97.156
Step: 9000 accuracy for training: 98.312  and testing 97.438
Step: 10000 accuracy for training: 98.219  and testing 98.031
Step: 19000 accuracy for training: 98.219  and testing 97.750

gates off:
Step: 0 accuracy for training: 10.688  and testing 10.344
Step: 1000 accuracy for training: 89.344  and testing 90.250
Step: 2000 accuracy for training: 92.188  and testing 92.031
Step: 3000 accuracy for training: 91.062  and testing 92.688
Step: 4000 accuracy for training: 92.969  and testing 93.125
Step: 5000 accuracy for training: 94.938  and testing 94.250
Step: 6000 accuracy for training: 94.594  and testing 95.281
Step: 7000 accuracy for training: 95.031  and testing 95.969
Step: 8000 accuracy for training: 95.281  and testing 95.688
Step: 9000 accuracy for training: 95.969  and testing 95.562
Step: 10000 accuracy for training: 96.062  and testing 95.656
Step: 11000 accuracy for training: 96.438  and testing 96.000
Step: 12000 accuracy for training: 95.625  and testing 95.344
Step: 13000 accuracy for training: 95.969  and testing 95.906
Step: 14000 accuracy for training: 96.625  and testing 97.344
Step: 15000 accuracy for training: 96.750  and testing 96.812
Step: 16000 accuracy for training: 97.031  and testing 96.500
Step: 17000 accuracy for training: 97.000  and testing 96.531
Step: 18000 accuracy for training: 97.062  and testing 96.312
"""

#accuracy with 1
#         train_accu = eval_model(train_images, train_labels, 1)
#         test_accu = eval_model(test_images, test_labels, 1)
        
#         print('1 iteration: accuracy for training: {:2.3f}  and testing {:2.3f}'.format(train_accu, test_accu ))
# #accuracy with 2 
#         train_accu = eval_model(train_images, train_labels, 2)
#         test_accu = eval_model(test_images, test_labels, 2)
        
#         print('2 iteration: accuracy for training: {:2.3f}  and testing {:2.3f}'.format(train_accu, test_accu ))




# def loss(model, x, y):
#   prediction = model(x)
#   return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets)
#   return tape.gradient(loss_value, model.variables)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# x, y = iter(dataset_train).next()
# print("Initial loss: {:.3f}".format(loss(model, x, y)))

# # Training loop
# for (i, (x, y)) in enumerate(dataset_train):
#   # Calculate derivatives of the input function with respect to its parameters.
#   grads = grad(model, x, y)
#   # Apply the gradient to the model
#   optimizer.apply_gradients(zip(grads, model.variables),
#                             global_step=tf.train.get_or_create_global_step())
#   if i % 200 == 0:
#     print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

# print("Final loss: {:.3f}".format(loss(model, x, y)))



# EPOCHS = 20

# for epoch in range(EPOCHS):
#     start = time.time()
#     total_loss = 0
    
#     for (batch, (img_tensor, target)) in enumerate(dataset):
#         loss = 0
        
#         dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
#         with tf.GradientTape() as tape:
#             features = encoder(img_tensor)
            
#             for i in range(1, target.shape[1]):
#                 # passing the features through the decoder
#                 predictions, hidden, _ = decoder(dec_input, features, hidden)

#                 loss += loss_function(target[:, i], predictions)
                
#                 # using teacher forcing
#                 dec_input = tf.expand_dims(target[:, i], 1)
        
#         total_loss += (loss / int(target.shape[1]))
#         variables = encoder.variables + decoder.variables
#         gradients = tape.gradient(loss, variables) 
#         optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        
#         if batch % 100 == 0:
#             print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
#                                                           batch, 
#                                                           loss.numpy() / int(target.shape[1])))
#     # storing the epoch end loss value to plot later
#     loss_plot.append(total_loss / len(cap_vector))
#     print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
#                                          total_loss/len(cap_vector)))
#     print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




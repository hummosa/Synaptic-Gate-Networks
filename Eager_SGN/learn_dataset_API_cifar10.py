# This is a short experiment to learn how to use the new TensorFlow Dataset object and an iterator to "effeciently" feed data into a model, but from this initial attempt 
# it takes MINUTES for the dataset objects to convert from np arrays and then another minute to load the iterator.
# Worse, it took almost all of the ram on my computer and crashed at some point.


import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
import sys
import os

os.chdir('D:\\code\\ai')
data_path = "/../data/cifar10/"

import cifar10
cifar10.maybe_download_and_extract()

(train_images, train_classes, train_labels) = cifar10.load_training_data()
(test_images, test_classes, test_labels)    = cifar10.load_test_data()
class_names = cifar10.load_class_names()

#Plot image func taken from hvass lab
def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name,
                                                       cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    print("Figure open, close it to continue.")
    plt.show()

plot_images(train_images[:9, :,:,:], train_classes[:9], smooth= True)


#convert into a tf.dataset directly from np arrays.

# dataset = tf.data.Dataset.from_tensor_slices( (train_images,train_labels) )
# dataset = dataset.batch(batch_size=9)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()

# sess = tf.Session()
# ex = sess.run(next_element)

# plot_images(ex[0], np.argmax(ex[1],1))

# sess.close()

# CONCLUSION: Super shitty. Takes lots of time to process, almost minutes, and took up the entire 8gb of ram I have! and crashed when I loaded more than several items.
# Wait for the next version 








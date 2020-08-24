import logging 
import os.path as osp 

import coloredlogs
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import datasets


coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


def imshow(image, label): 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(image, cmap=None)
    plt.axis("off")
    plt.title(label)
    plt.show()
    plt.close(fig)




if __name__ == "__main__":

    
    # === Input Image Preprocessing === # 
    """ input feature visualization 
    1. Load image dataset 
    2. Visualize them 

    required package:   os 
                        glob
                        matplotlib 
                        tensorflow.keras.datasets 
    """
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    img, label = train_x[0], train_y[0]
    logging.info(f"dataset shape: {train_x.shape}")  # Batch x Heigh x Width 
    logging.info(f"image shape: {img.shape}")
    imshow(image=img, label=label)

    """
    Given MNIST-image shape is (28, 28). 
    However, it's not suitable for CNN. 
    It should have [batch, height, width, channel] shape for tensorflow Conv-layer 
    """
    image_batch = img[tf.newaxis, ..., tf.newaxis]
    logging.info(f"batch data: {image_batch.shape}")

    # === Feature Extraction === # 
    """
    1. convolution parameters:   
                                -filters : how many filters are made out of layers (a.k.a., weights, filters, channels)
                                -kernel_size : filter(weight) size 
                                -strides : how many pixels are skipped when the filter passing over
                                -padding : zero padding, VALID(=no padding), SAME(=same size like before)
                                -activation : Activation Function (= It can be made in sperated layer)
    
    Usage : tf.keras.layers.Conv2D()


    2. pooling (= subsampling)
    """
    tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')
    

    """ 
    filter visualization 
    """
    image = tf.cast(image_batch, dtype=tf.float32) # for convolution operation 
    layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu')
    output = layer(image)

    logging.info(f"output feature_maps : ")
    logging.info(output)

    imshow(image=output[0, :, :,3], label="feature_map")

    

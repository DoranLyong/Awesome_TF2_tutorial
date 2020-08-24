import logging 
import os.path as osp 

import coloredlogs 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import datasets, layers 

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


def imshow(image, label): 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(image, cmap=None)
    plt.axis("off")
    plt.title(label)
    plt.show()
    plt.close(fig)



# Hyperparameters 
input_shape = (28, 28, 1)
num_classes = 10 

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

    # === Feature Extraction === # 
    """ CNN block-1 """
    inputs = layers.Input(shape=input_shape)

    net = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="SAME")(inputs)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="SAME")(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(net)
    net = layers.Dropout(0.25)(net) # preserve 25% nodes 

    """ CNN block-2 """
    net = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME")(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME")(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(net)
    net = layers.Dropout(0.25)(net) # preserve 25% nodes 

    # === Fully-Connected === # 
    net = layers.Flatten()(net)
    net = layers.Dense(512)(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.25)(net)
    net = layers.Dense(10)(net)    # 10 classes 
    net = layers.Activation('softmax')(net)


    # === Model init. === # 
    model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

    
    # === Model summary === # 
    model.summary()
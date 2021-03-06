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

def featuremap_show(output, weight): 
    plt.figure(figsize=(15,3))
    
    plt.subplot(121)
    plt.title(weight.shape)
    plt.imshow(weight[:, :, 0, 0], cmap=None)

    plt.subplot(122)
    plt.title(output.shape)
    plt.imshow(output[0, :, :, 0], cmap=None)
    plt.colorbar()
    plt.show()
    plt.close()
    
def hist_show(output, title):
    plt.figure(figsize=(15,3))
    plt.hist(output.numpy().ravel(), range=(np.min(output), np.max(output)), edgecolor='black')
    plt.title(title)
    plt.show()

def vis_flatten(flatten, title):
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.hist(flatten.numpy().ravel())
    
    plt.subplot(212)
    plt.imshow(flatten[:,:100], 'jet')
    plt.title(title)
    plt.show()




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

    3. fully-connected layer (=dense layer)

    4. Flatten 

    5. Dropout

    """
    tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')
    

    """ 
    filter visualization 
    """
    image_batch = tf.cast(image_batch, dtype=tf.float32) # for convolution operation 
    layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu')
    output = layer(image_batch)

    logging.info(f"output feature_maps : ")
    logging.info(output)

    imshow(image=output[0, :, :,0], label="feature_map")

    logging.info(f"min-max range of input:{np.min(image_batch)}, {np.max(image_batch)}")
    logging.info(f"min-max range of feature_map:{np.min(output[0,:,:,0])}, {np.max(output[0,:,:,0])}")


    """
    To get weights(=filter) : 
                                - layer.get_weights() 
    """
    weight, bias = layer.get_weights()
    logging.info(f"weight_shape, bias_shape : {weight.shape}, {bias.shape}")
    featuremap_show(output=output, weight=weight)
    hist_show(output=output, title="hist. of the output")
    

    # === Activate function === # 
    """
    Activate layer can be defined in the seperated one. : 

                                                            - tf.keras.layers.ReLU()
                                                            - tf.keras.layers.Activation('relu')
    """
    conv2d_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='SAME', activation="relu" )
    act_layer = tf.keras.layers.ReLU()

    act_output = act_layer(conv2d_layer(image_batch))  # input -> conv2d -> activate 
    logging.info(act_output.shape)
    logging.info(f"feature_map min, max: {np.min(act_output)}, {np.max(act_output)}")


    # === Pooling (= subsampling) === # 
    """
    tf.keras.layers.MaxPool2D() parameters: 
                                            - pool_size 
                                            - strides 
                                            - padding 
    """
    pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")
    pool_output = pool_layer(output)
    logging.info(f"MaxPooling feature map : {pool_output.shape}")

    hist_show(output=pool_output, title="MaxPooling")
    imshow(image=pool_output[0,:,:,0], label="MaxPooling")


    # === Flatten === # 
    """
    tf.keras.layers.Flatten() 
    """
    flat_layer = tf.keras.layers.Flatten() 
    flatten = flat_layer(output)
    logging.info(f"shape before flatten: {output.shape}") # Batch x Height x Width x Channel
    logging.info(f"flatten shape of the output: {flatten.shape}") # Batch x flat_vector
    vis_flatten(flatten=flatten, title="flatten")


    # === Dense (=fully-connected) === # 
    """
    tf.keras.layers.Dense(): 
                                - units : how many output nodes? 
                                - activation

    """
    dense_layer = tf.keras.layers.Dense(32, activation="relu")
    dense_output = dense_layer(flatten)
    
    logging.info(f"dense output: {dense_output.shape}")


    # === Dropout === # 
    """
    tf.keras.layers.Dropout()
    """
    drop_layer = tf.keras.layers.Dropout(0.7) # save 70% nodes 
    drop_output = drop_layer(dense_output)

    logging.info(f"dropout : {drop_output.shape}")
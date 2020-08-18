import logging

import coloredlogs
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


def imshow(img, label): 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(img, cmap=None)
    plt.axis("off")    
    plt.title(label)    
    plt.show()
    plt.close(fig)



if __name__ =="__main__":
    
    # == Data Loading === # 
    """ Load built-in MNIST dataset:    
    from tensorflow.keras import datasets   
    """
    mnist = datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    logging.info(f"input_feature shape: {train_x.shape}")  # Batch x Heigh x Width
    logging.info(f"label shape: {train_y.shape}")

    
    # == Image Dataset Visualize == # 
    """
    1. take one image data from the batch 
    2. Visualize it 
    """
    img = train_x[0] 
    logging.info(f"image shape: {img.shape}")
    logging.info(f"label dtype: {train_y[0].dtype}")

    imshow(train_x[0], train_y[0])

    
    # == Data Channel shape == # 
    """ 
    tf.Tensor's array order  :
                                [Batch, Heigh, Width, Channel]
    1. expand dimensions : 
                            with 'numpy' method  : np.expand_dims(<array>, dim)
                            with 'tf' method : tf.expand_dims(<Tensor>, dim)
                                               tf.newaxis 
    """
    given_data = train_x
    logging.info(f"check given data shape: {given_data.shape}")

    expanded_data = np.expand_dims(given_data, -1)    # -1 : add dim at the end 
    logging.info(f"expanded dim at the end axis: {expanded_data.shape}")
    logging.info(f"expanded dim at 0th axis: {np.expand_dims(given_data, 0).shape}")

    expanded_tensor = tf.expand_dims(given_data, -1)
    logging.info(f"expanded dim of Tensor at the end axis : {expanded_tensor.shape}")
    logging.info(given_data[...,tf.newaxis].shape)   # by official homage
    logging.info(given_data.shape)

    print()

    """
    2. reduce dimensions :  
                            for imshow 
    if given gray-image has [28, 28, 1] dimension, 
    it should be reshape as [28, 28]

                                        : explicite with indexing 
                                        : np.squeeze()   reduce the single dimension
    """
    new_train_x = train_x[..., tf.newaxis] 
    logging.info(f"Before reducing dimensions: {new_train_x.shape}")

    disp = new_train_x[0]
    logging.info(f"This shape of gray image makes error for visualization: {disp.shape}")  # [28, 28, 1]
    logging.info(disp[:,:,0].shape)  # explicite one channel element 

    logging.info(f"dtype of the image: {type(disp)}")  # the loaded image is <nparray>
    logging.info(f"squeeze the sigle dim: {np.squeeze(disp).shape}")
    imshow(np.squeeze(disp), 'reduced_dims')

    print()

    # === Label Dataset Analysis === # 

    logging.info(f"train_y shape : {train_y.shape} ")
    logging.info(f"0th label data: {train_y[0]}")
    logging.info(f"dype of it : {train_y[0].dtype}")

    

    # === One-Hot Encoding === # 
    """ ex_ Classification task for 5 classes : [0, 0, 0, 1, 0] or [1, 0, 0, 0, 0] etc. 

    1. simple_method :  tensorflow.keras.utils.to_categorical 
    """

    onehot4one = to_categorical(y=1, num_classes=10)
    logging.info(f"OneHot encoding: {onehot4one}")

    label = train_y[0]  
    onehot_label = to_categorical(y=label, num_classes=10)
    logging.info(f"label : {label}")
    logging.info(f"onehot_label : {onehot_label}")


    imshow(image = train_x[0], label = onehot_label)
    



    







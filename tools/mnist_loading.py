import logging

import coloredlogs
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import datasets

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


def imshow(img, label): 
    fig, ax = plt.subplots(figsize=(8,8))
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
    imshow(train_x[0], train_y[0])




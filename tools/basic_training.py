import logging 
import os.path as osp 

import coloredlogs 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import datasets
from tensorflow.keras import layers 

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
num_epochs = 1
batch_size = 32 


if __name__ == "__main__":
    
    # === Prepare MNIST Dataset=== # 
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


    # === Build Model === # 
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

    """ Fully-Connected """
    net = layers.Flatten()(net)
    net = layers.Dense(512)(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.25)(net)
    net = layers.Dense(10)(net)    # 10 classes 
    net = layers.Activation('softmax')(net)


    """ Model init. """ 
    model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
    model.summary() # model summary show 


    # === Optimization cfg === # 
    """
    Configurate before training: 
    1. Loss function 
    2. Metrics 
    3. Optimization method 
    
    """


    """
    Loss Function : 
                    - Binary cross-entropy 
                    - Categorical cross-entropy 
                    - Sparse categorical cross-entropy 
    """
#    loss = 'binary_crossentropy'
#    loss = 'categorical_crossentropy'

#    tf.keras.losses.binary_crossentropy
#    tf.keras.losses.categorical_crossentropy
    
    loss_func = tf.keras.losses.sparse_categorical_crossentropy


    """
    Metrics : model estimation 

    - accuracy 
    - tf.keras.metrics 
    """
#    tf.keras.metrics.Accuracy()
#    tf.keras.metrics.Precision()
#    tf.keras.metrics.Recall()
    
    metrics = ['accuracy']


    """
    Optimization method : 
                            - 'sgd' 
                            - 'rmsprop'
                            - 'adam'
    """
#    tf.keras.optimizers.SGD()
#    tf.keras.optimizers.RMSprop()
    optim = tf.keras.optimizers.Adam()    


    # === Compile === #
    model.compile(optimizer= optim, loss= loss_func, metrics= metrics )



    # === Prepare Dataset === # 
    """
    Check shape 

    For inserting your data into the network,
    your data shape should be (Batch, Height, Width, Channel).
    """
    logging.info(f"train input, label shape: {train_x.shape}, {train_y.shape}")
    logging.info(f"test input, label shape: {test_x.shape}, {test_y.shape}")

    train_x = train_x[ ... , tf.newaxis]
    test_x = test_x[ ... , tf.newaxis]

    logging.info(f"train input, label shape: {train_x.shape}, {train_y.shape}")
    logging.info(f"test input, label shape: {test_x.shape}, {test_y.shape}")
    

    """
    Rescaling -> normalization between 0 and 1 
    """
    logging.info(f"value min-max : {np.min(train_x)}, {np.max(train_x)}")
    
    train_x = train_x / np.max(train_x).astype(np.float64)
    test_x = test_x / np.max(test_x).astype(np.float64)

    logging.info(f"rescaled min-max : {np.min(train_x)}, {np.max(train_x)}")


    # === Training === # 
    """
    1. Configurate Hyperparameters :
                                    - num_epochs 
                                    - batch_size 
                                    - etc. 

    2. model.fit() 
    """
    model.fit(train_x, train_y,
              batch_size = batch_size, 
              shuffle= True, 
              epochs=num_epochs
            )


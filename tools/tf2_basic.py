import logging 

import coloredlogs
import numpy as np 
import tensorflow as tf 


coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")



if __name__ == "__main__": 
    
    # === generate Tensor === # 
    """
    [], () 
    np.array  
    tf.tensor
    """
    t_list = [1, 2, 3]
    t_tuple = (1, 2, 3)

    arr_1D = np.array([1,2,3])
    arr_2D = np.array([[1, 2, 3], [4, 5, 6]])   # list in list 
    logging.info(f"arr_1D shape: {arr_1D.shape}") 
    logging.info(f"arr_2D shape: {arr_2D.shape}")

    """
    tf.constant()  : list -> Tensor 
                   : tuple -> Tensor 
                   : array -> Tensor 
    """
    logging.info(tf.constant([1, 2, 3]))
    logging.info(tf.constant(((1,2,3),(4,5,6))))
    logging.info(tf.constant(np.array([[1, 2, 3,], [4,5,6]])))

    """
    Check tensor attributes:  shape 
                              data type (check, def, convert) 
                              Tensor -> nparry 
    """
    tensor = tf.constant(np.array([[1, 2, 3,], [4,5,6]]))
    logging.info(f"Tensor shape: {tensor.shape}")
    logging.info(f"Tensor dtype: {tensor.dtype}")         # dtype check

    float_arr = np.array([1, 2, 3,], dtype=np.float32)    # dtype def 
    uint8_arr = float_arr.astype(np.uint8)                # dtype cvt 

    float_tensor = tf.constant([1,2,3], dtype=tf.float32) # dtype def 
    uint8_tensor = tf.cast(float_tensor, dtype=tf.uint8)  # dtype cvt 
    logging.info(float_tensor)
    logging.info(uint8_tensor)

    logging.info(f"tf.Tensor -> np.array: {tensor.numpy()}")     # tf.Tensor -> np.array 
    logging.info(f"tf.Tensor -> np.array: {np.array(tensor)}")
    logging.info(f"Check: {type(tensor.numpy())}")
    logging.info(f"Check: {type(tensor)}")

    
    # === Generate Random Numbers === # 
    """
    Uniform distribution 
    Normal distribution 
    """
    logging.info(f"4 elements normal random: {np.random.randn(4)}")     # normal distribution in np.array 
    logging.info(f"normal random around mean 4: {np.random.normal(4)}")    
    print("\n")

    logging.info(f"4 elements normal random vector: {tf.random.normal([4,])}") 
    logging.info(f"2x2 elements normal random matrix: {tf.random.normal([2,2])}") 
    logging.info(f"4 elements uniform random vector: {tf.random.uniform([4,])}")


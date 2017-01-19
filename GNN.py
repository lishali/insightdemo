import tensorflow as tf
import numpy as np
import scipy as sp


def batch_norm_wrapper(inputs, batch_mean=None, batch_var=None, beta=None, scale=None, epsilon=None, is_training=True):
    
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        mean, var = tf.nn.moments(inputs,[0])
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
    
def layer_accuracy(input_vector,
                   filter_touse, 
                   beta, scale,
                   batch_size, 
        signal_dim_in, signal_dim_out, channel_dim, SD, 
          cascade_list, dim,
          XY,
         filter_name=None, 
         scale_name=None, 
         beta_name=None, 
         relu=True):
    """A layer of the GNN to evaluate accuracy
    We assume that signal_dim_in does not include the dirac signal and degree signal"""
    cascade_2 = [tf.batch_matmul(cascade_list[i], input_vector[j]) for j in xrange(signal_dim_in+2) for i in xrange(channel_dim)] 
    cascade_2_reshape = tf.reshape(cascade_2, shape=[signal_dim_in+2, channel_dim, batch_size, dim])
    input_2 = tf.transpose(cascade_2_reshape, [2, 1, 3, 0])
    op2 = tf.nn.conv2d(input_2, filter_touse, strides=[1, 1, 1, 1], padding='VALID')
    batch_mean2, batch_var2 = tf.nn.moments(op2, [0])
    BN2 = batch_norm_wrapper(op2, batch_mean2, batch_var2, 
                                   beta, scale, epsilon)
    if relu:
        BN2 = tf.nn.relu(BN2)
        
    zc2 = tf.transpose(tf.concat(3, [XY, BN2]), perm=[3,0,2,1])
    return zc2


def preprocess(batch_size,
              Adj, Y_prepare):


    A = tf.cast(Adj, dtype=tf.float32)
    D = tf.reduce_sum(A, 1)
    D_sqrt = tf.sqrt(D)
    D_sqrt_inv = tf.matrix_diag(tf.reciprocal(D_sqrt))

    Iden = tf.reshape(tf.concat(0, [tf.Variable(np.identity(dim), dtype=tf.float32, trainable=False)]*batch_size),
                       shape=[batch_size, dim, dim])
    Q = tf.batch_matmul(tf.batch_matmul(D_sqrt_inv, A), D_sqrt_inv)#tf.matmul(D, Adj, a_is_sparse=True)
    Q1 = tf.batch_matmul(Q, Q)
    Q2 = tf.batch_matmul(Q1, Q1)
    Q3 = tf.batch_matmul(Q2, Q2) 

    cascade_lst = [Iden, Q, Q1, Q2, Q3]
    channel_dim = len(cascade_lst)

    X = tf.expand_dims(D, 2)

    Y = tf.expand_dims(Y_prepare, 2)

    XY = tf.concat(0, [tf.expand_dims(X, 0), tf.expand_dims(Y, 0)])

    xy = tf.transpose(XY, perm=[1, 3, 2, 0])
    
    return XY, xy, cascade_lst



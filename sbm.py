import tensorflow as tf
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
from dplython import *

def Diffusion(A, F):
    """finds the diffusion of F via A"""
    #F can be a batched signal
    return tf.batch_matmul(A, F)

def Diag(A, F):
    """multiplies F by diagonal vector"""
    diag_matrix = tf.expand_dims(tf.reduce_sum(Adj, 1), 1)
    return tf.mul(diag_matrix, F)

def Q_op(A, F):
    """D^-1W"""
    return tf.mul(tf.div(1.0, Diag(A, F)), tf.batch_matmul(A, F))


def balanced_stochastic_blockmodel(big_community, small_community, p_in=1.0, p_out=0.0, seed=None):
    """gives dense adjacency matrix representaiton of randomly generated SBM with balanced community size"""

    G = nx.random_partition_graph([big_community, small_community], p_in=p_in, p_out=p_out, seed=seed)
    A = nx.adjacency_matrix(G).todense()
    
    return A

def expand_dim(lst):
    return(tf.concat(2, [tf.expand_dims(lst[i], 2) for i in xrange(len(lst))]))

def cascade(signal, Q_lst):
    return [tf.matmul(Q_lst[i], signal) for i in xrange(len(Q_lst))]


def convolution_block(Name, input_signal, output_channel_dim, input_channel_dim, signal_dim, batch_size=1, SD=1):
    
    input_1 = tf.expand_dims(input_signal, 0)
    filter_1 = tf.Variable(tf.random_normal([batch_size, signal_dim, input_channel_dim, output_channel_dim], stddev=SD), 
                          name=Name, 
                         trainable=True)
    return(filter_1, tf.nn.conv2d(input_1, filter_1, strides=[1, 1, 1, 1], padding='VALID'))

def parameters_data(p = 0.2, q=0.02, big_community=200, small_community=48, batch_size=1, datapoints = 100, data_rep=10):

    DATA1 = [np.asarray(balanced_stochastic_blockmodel(big_community=big_community, 
                                                       small_community=small_community,
                                                       p_in=p, p_out=q, seed=None)).astype(np.double) for k in xrange(datapoints)]
    DATA = DATA1*(data_rep)
    np.random.shuffle(DATA)

    TRUE_A = np.append(np.ones([batch_size, big_community], dtype=float),
                       np.zeros([batch_size, small_community], dtype=float), axis = 1)

    return(DATA, TRUE_A)

def SBM(p_in=1, p_out=0, big_community=2, small_community=2, directed=False):
    dim = big_community+small_community
    
    perms = [i for i in xrange(dim)]
    np.random.shuffle(perms)
    
    G = nx.random_partition_graph([big_community,small_community], p_in=p_in, p_out=p_out, directed=directed)
    A = nx.to_numpy_matrix(G, perms)

    truth_setup = [1.0]*big_community+[-1.0]*small_community
    true_label = [truth_setup[i] for i in perms]
    
    dirac_setup = [1.0]+[0.0]*(dim-1)
    true_dirac = [dirac_setup[i] for i in perms]
    
    return A, true_label, true_dirac, perms

def DATA_SBM(p_in=1, p_out=0, big_community=2, small_community=2, data_points=10, data_rep=1, directed=False):
    Data1 = [SBM(p_in=p_in, p_out=p_out, big_community=big_community, small_community=small_community, directed=directed) for i in xrange(data_points)]
    
    np.random.shuffle(Data1)
    
    Data = Data1*data_rep
    
    return Data 
    

# Layer wrappers for tensorflow
from typing import List, Any, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras.engine.base_layer import DeferredTensor

he_normal = tf.keras.initializers.he_normal()
he_uniform = tf.keras.initializers.he_uniform()
constant = tf.constant_initializer(0.0)
random_normal = tf.random_normal_initializer(stddev=0.02)

def lrelu(x,alpha=0.2,name=None):
    x = tf.nn.leaky_relu(x,alpha,name)
    return x

def relu(x,name=None):
    x = tf.nn.relu(x,name=name)
    return x

def tanh(x,name=None):
    x = tf.nn.tanh(x,name=name)
    return x

def sigmoid(x,name=None):
    x = tf.nn.sigmoid(x,name=name)
    return x

def softmax(x,name=None):
    x = tf.nn.softmax(x,name=name)
    return x

def dropout(x,prob,name=None):
    x = tf.nn.dropout(x,prob,name=name)
    return x

def flatten(x,name=None):
    x = tf.layers.flatten(x, name=name)
    return x

def batchNorm(x,axis=-1,momentum=0.99,epsilon=0.001,name=None):
    x = tf.layers.batch_normalization(x,axis,momentum,epsilon,name)
    return x

def instNorm(x, name=None):
    x = tf.contrib.layers.instance_norm(x, scope=name)
    return x

def groupNorm(x, group=32, name=None):
    x = tf.contrib.layers.group_norm(x, groups=group, scope=name)
    return x

def pixNorm(x,eps=1e-8,name=None):
    with tf.variable_scope(name):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x),axis=1,keepdims=True)+eps)

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(axis=3, num_or_size_splits=3, value=X)
        X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X

def fullConn(x,units,init=None,name=None):
    out = tf.layers.dense(x,units,kernel_initializer=init, name=name)
    return out

def conv(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    out = tf.layers.conv2d(x,f,ksz,s,pad,kernel_initializer=init,name=name)
    return out

def convBN(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = conv(x,f,ksz,s,pad,init=init,name='conv')
        out = batchNorm(out,name='BN')
        return out

def convLrelu(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = conv(x,f,ksz,s,pad,init=init,name='conv')
        out = lrelu(out,name='lrelu')
        return out

def convBNLrelu(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = conv(x,f,ksz,s,pad,init=init,name='conv')
        out = batchNorm(out,name='BN')
        out = lrelu(out,name='lrelu')
        return out

def deconv(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    out = tf.layers.conv2d_transpose(x,f,ksz,s,pad,kernel_initializer=init,name=name)
    return out

def deconvBN(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = deconv(x,f,ksz,s,pad,init=init,name='deconv')
        out = batchNorm(out, name='lrelu')
        return out

def deconvBNLrelu(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = deconv(x,f,ksz,s,pad,init=init,name='deconv')
        out = batchNorm(out, name='BN')
        out = lrelu(out, name='lrelu')
        return out

def deconvLrelu(x,f,ksz=3,s=2,pad='same',init=None,name=None):
    with tf.variable_scope(name):
        out = deconv(x,f,ksz,s,pad,init=init,name='deconv')
        out = lrelu(out, name='lrelu')
        return out

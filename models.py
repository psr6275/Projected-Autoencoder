import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.distributions as dist

slim = tf.contrib.slim

def AE_dense(X=None,dims = [784,1024,1024],activation =tf.nn.elu,tied=False):
    """
    Build a deep denoising autoencoder w/ or w/o tied weight

    Parameters
    ----------


    Returns
    -------
    """
    shapeX = X.shape
    #print(shapeX)
    with tf.variable_scope("AE_dense") as vs:
        #Encoder
        z = tf.reshape(X, [-1, np.prod(shapeX[1:])]) # At first, you need to flaten your data :)
        for idx, n_output in enumerate(dims[1:]):
            z = slim.fully_connected(z,n_output,activation_fn=activation)
            z = slim.batch_norm(z)
        #Decoder
        rx = z
        for idx, n_output in enumerate(dims[:-1][::-1]):
            if idx < len(dims)-2:
                rx = slim.fully_connected(rx,n_output,activation_fn=activation)
                rx = slim.batch_norm(rx)
            else:
                rx = slim.fully_connected(rx,n_output,activation_fn = tf.nn.sigmoid)
    variables = tf.contrib.framework.get_variables(vs)
    return rx, z, variables

def AE_conv(X = None, filters =[3,64,128,32,1024] ,activation = tf.nn.elu,data_format = "NHWC"):
    shapes = []
    shapes.append(int_shape(X))
    shapeX = X.shape
    if data_format =="gray":
        X = tf.reshape(X,[-1,shapeX[1],shapeX[2],1])
        filters[0] = 1
    with tf.variable_scope("AE_dense") as vs:
        z = X
        for idx, n_output in enumerate(filters[1:-1]):
            z = slim.conv2d(z,n_output,3,1,activation_fn = activation)
            z = slim.conv2d(z,n_output,3,1,activation_fn = activation)
            z = slim.batch_norm(z)
            z = slim.conv2d(z, n_output,3,2,activation_fn = activation)
            shapes.append(int_shape(z))
        shapeZ = int_shape(z)        
        print("print(shapeZ): ",shapeZ)
        z = tf.reshape(z,[-1,np.prod(shapeZ[1:])])
        print("print(reshaped_z): ",int_shape(z))
        z = slim.fully_connected(z,filters[-1],activation_fn = None)
        print("filters: ",filters[-1])
        print("print(after_fully_connected_z): : ",int_shape(z))
        #Decoder
        print(type(np.prod(shapeZ[1:])))
        rx = slim.fully_connected(z,int(np.prod(shapeZ[1:])))
        #rx = z
        print("print(rx): ",int_shape(rx))
        rx = tf.reshape(rx,[-1,shapeZ[1],shapeZ[2],shapeZ[3]])
        shapes = shapes[::-1]
        for idx, n_output in enumerate(filters[1:-1][::-1]):
            #print(n_output)
            #print(type(n_output))
            rx = slim.conv2d(rx,n_output,3,1,activation_fn = activation)
            rx = slim.conv2d(rx,n_output,3,1,activation_fn = activation)
            rx = slim.batch_norm(rx)
            rx = tf.image.resize_nearest_neighbor(rx,(shapes[idx+1][1],shapes[idx+1][2]))
            #rx = upscale(rx,2)
        rx = slim.conv2d(rx,filters[0],3,1,activation_fn = None)
    print("shape of the final rx: ",int_shape(rx))
    if data_format =="gray":
        rx = tf.reshape(rx,[-1,shapeX[1],shapeX[2]])
        print("shape of the final rx in case of gray",int_shape(rx))

    variables = tf.contrib.framework.get_variables(vs)
    return rx, z, variables
def int_shape(tensor):
    shape=tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    if data_format == 'NCHW':
        return [shape[0],shape[2],shape[3],shape[1]]
    elif data_format =='NHWC':
        return shape

def resize_nearest_neighbor(x,new_size):
    return tf.image.resize_nearest_neighbor(x,new_size)

def upscale(x,scale):
    #_, h,w, _  = get_conv_shape(x,data_format)
    _,h,w,_ = int_shape(x)
    return resize_nearest_neighbor(x, (h*scale, w*scale))

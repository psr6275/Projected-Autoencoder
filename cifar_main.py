from keras.layers import Input, Dense,Dropout, Flatten,Lambda
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split as tts
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.utils import np_utils,plot_model
from keras.layers.convolutional import Convolution2D,MaxPooling2D,Conv2D
from keras import backend as K
from keras.losses import categorical_hinge,categorical_crossentropy
import tensorflow as tf

from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print(x_train.shape)

Xtr,Xval,Ytr,Yval = tts(x_train,Y_train,test_size = 0.3)

import sys
sys.path.append('../')
from cifar_clf import *
from cifar_tfdae import *
print("load function and library")

cifar_dae = Cifar10_DAE(x_train,Y_train,num_epoch=100,loss_type = "l1")
cifar_dae.train()

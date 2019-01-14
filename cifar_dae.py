from keras.layers import Input, Dense,Dropout, Flatten, Lambda,Layer
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Conv2DTranspose
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
#from sklearn.model_selection import train_test_split as tts
#from keras.utils import np_utils, plot_model
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras import backend as K
#from keras.losses import categorical_hinge, categorical_crossentropy
import numpy as np
from mnist_ds import *
import matplotlib.pyplot as plt
from mnist_dae import corrupt

def cifar_cdae(filter_nums = [32,64]):
    """convolutional dae with undercomplete structure"""


    input_img = Input(shape = (32,32,3))
    encoded = input_img
    for flt in filter_nums:
        encoded = Conv2D(flt, (3,3), padding='same')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)

    decoded = encoded
    for flt in filter_nums[::-1]:
        decoded = Conv2DTranspose(flt,(3,3),padding='same')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2DTranspose(3,(3,3),padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('sigmoid')(decoded)


    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    # no decoder in case of convolution_dae!
    return encoder, autoencoder

def cifar_cdae_over(filter_nums = [32,64]):
    """convolutional dae with undercomplete structure"""


    #input_img = Input(shape = (32,32,3))
    #encoded = input_img
    """
    for flt in filter_nums:
        encoded = Conv2DTranspose(flt, (3,3), strides = 2,padding='valid')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        encoded = UpSampling2D((2,2))(encoded)
        #encoded = MaxPooling2D((2,2),padding='same')(encoded)

    decoded = encoded
    for flt in filter_nums[::-1]:
        decoded = Conv2D(flt,(3,3),padding='same')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)
        decoded = MaxPooling2D((2,2),padding='same')(decoded)
        #decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2D(3,(3,3),padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('sigmoid')(decoded)
    """
    """
    ##lsy
    autoencoder = Sequential([
    Conv2DTranspose(2,(5,5),strides=2,padding='valid',input_shape=(32,32,3)),
    Conv2DTranspose(1,(5,5), strides =2, padding='valid'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(2,(4,4),padding='valid'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Activation('relu'),

    Conv2D(3,(4,4),padding='valid'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Activation('sigmoid')
        ])
    """
    ## srcnn (super-resolution)
    #autoencoder = Sequential([
    input_img = Input(shape = (32,32,3))
    encoded = Conv2D(64,(9,9),padding = 'same', activation = 'relu')(input_img)
    encoded = Conv2D(32,(1,1),padding = 'same',activation ='relu')(encoded)
    encoded = Conv2D(3,(5,5),padding = 'same',activation = 'sigmoid')(encoded)
    #encoded = K.clip(encoded,0,1)
    #])

    #encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, encoded)
    # no decoder in case of convolution_dae!
    return autoencoder

class Cifar_DAE:
    def __init__(self,trainX,trainY,filter_nums = [32,64],num_batch = 128,test_size = 0.3,dae_type = "over",
                 loss_type = "mean_squared_error",noise_type = 'gaussian',noise_scale = 0.3,epoch=30):
        self.data = trainX
        if trainY is not None:
            self.dataY = trainY
        self.filter_nums = filter_nums
        self.num_batch = num_batch
        self.test_size = test_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.epoch = epoch
        self.loss_type = loss_type
        self.dae_type = dae_type
        self.build_dae()


    def build_dae(self):
        if self.dae_type == "over":
            self.autoencoder = cifar_cdae_over()
        else:
            self.encoder, self.autoencoder = cifar_cdae(self.filter_nums)
    def train_dae(self):
        self.autoencoder.compile(optimizer = 'adadelta',loss=self.loss_type)
        self.trainX, self.trainXn = corrupt(self.data,noise_type = self.noise_type,scale = self.noise_scale)
        self.idxs = np.array(range(len(self.data)))
        np.random.shuffle(self.idxs)
        val_num = int(len(self.idxs)*self.test_size)
        xtr_o = self.trainX[self.idxs[val_num:]]
        xtr_n = self.trainXn[self.idxs[val_num:]]
        xval_o = self.trainX[self.idxs[:val_num]]
        xval_n = self.trainXn[self.idxs[:val_num]]
        print("loss type is",self.loss_type)
        print("dae network is", self.dae_type)
        print("train the DAE model with noise", self.noise_type, " (", self.noise_scale, ")")
        self.autoencoder.fit(xtr_n,xtr_o,epochs = self.epoch,batch_size = self.num_batch,
                             shuffle = True,validation_data = (xval_n,xval_o),
                             callbacks =[TensorBoard(log_dir = '../logs/mnist_denseDAE',
                                                     histogram_freq=0, write_graph=False)])
    def plot_imgs(self,testX,noise_type = 'peppSalt',noise_scale = 0.3):
        xtest_o, xtest_n = corrupt(testX, scale = noise_scale, noise_type = noise_type)
        decoded_imgs = self.autoencoder.predict(xtest_n)
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(xtest_n[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display Reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

    ## dynamical system using the trained DAE model
    def apply_DS(self, testX, vr=0.9, max_iter=30):
        revX = self.autoencoder.predict(testX)
        projX = vr*testX + (1-vr)*revX
        for i in range(max_iter):
            revX = self.autoencoder.predict(projX)
            projX = vr*projX +(1-vr)*revX

        return projX

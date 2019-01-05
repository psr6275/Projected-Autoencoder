from keras.layers import Input, Dense,Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Conv2DTranspose
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
#from sklearn.model_selection import train_test_split as tts
#from keras.utils import np_utils, plot_model
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
#from keras import backend as K
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

class Cifar_DAE:
    def __init__(self,trainX,trainY,filter_nums = [32,64],num_batch = 128,test_size = 0.3,noise_type = 'gaussian',noise_scale = 0.3,epoch=30):
        self.data = trainX
        if trainY is not None:
            self.dataY = trainY
        self.filter_nums = filter_nums
        self.num_batch = num_batch
        self.test_size = test_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.epoch = epoch
        self.build_dae()

    def build_dae(self):
        self.encoder, self.autoencoder = cifar_cdae(self.filter_nums)
    def train_dae(self):
        self.autoencoder.compile(optimizer = 'adadelta',loss='binary_crossentropy')
        self.trainX, self.trainXn = corrupt(self.data,noise_type = self.noise_type,scale = self.noise_scale)
        self.idxs = np.array(range(len(self.data)))
        np.random.shuffle(self.idxs)
        val_num = int(len(self.idxs)*self.test_size)
        xtr_o = self.trainX[self.idxs[val_num:]]
        xtr_n = self.trainXn[self.idxs[val_num:]]
        xval_o = self.trainX[self.idxs[:val_num]]
        xval_n = self.trainXn[self.idxs[:val_num]]

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
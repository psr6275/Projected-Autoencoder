from keras.layers import Input, Dense,Dropout, Flatten, Lambda
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

def corrupt(x,scale=0.5, rep =1, noise_type = 'gaussian'):
    x_rep = np.repeat(x,rep,axis=0)
    if noise_type == 'gaussian':
        noise = np.random.normal(size = x_rep.shape)
        x_crr = x_rep+scale*noise
    elif noise_type =='corruption':
        noise = np.random.binomial(1,1-scale,size = x_rep.shape)
        x_crr = x_rep * noise
    elif noise_type =='peppSalt':
        noise = np.random.choice([-1,0,1],size = x_rep.shape,
                                   p = [scale/2,1-scale,scale/2])
        x_crr = x_rep + noise
    return np.clip(x_rep,0.0,1.0), np.clip(x_crr,0.0,1.0)

def mnist_dae(dims = [784,1024,2048]):
    """

    :param dims (the list of dimensoins for hidden layer):

    :return encoder (encoder model for dae)
            decoder (encoder part of dae model)
            autoencoder (reconstruction function of dae model)
    """
    input_img = Input(shape = (dims[0],))
    encoded = input_img
    for idx, dim in enumerate(dims[1:]):
        encoded = Dense(dim, activation='elu')(encoded)

    decoded = encoded
    for idx, dim in enumerate(dims[:-1][::-1]):
        if idx < len(dims)-2:
            decoded = Dense(dim,activation='elu')(decoded)
        else:
            decoded = Dense(dim,activation='sigmoid')(decoded)
    encoder = Model(input_img,encoded)
    autoencoder = Model(input_img,decoded)
    input_z = Input(shape=(dims[-1],))
    decoder_layers = autoencoder.layers[len(dims):]
    z_decoded = input_z
    for lyr in decoder_layers:
        z_decoded = lyr(z_decoded)
    decoder = Model(input_z,z_decoded)
    return encoder, decoder, autoencoder

class Mnist_DAE:
    def __init__(self,trainX,trainY,dims = [784,1024,2048],num_batch = 128,test_size = 0.3,noise_type = 'gaussian',noise_scale = 0.3,epoch=30):
        self.data = trainX
        if trainY is not None:
            self.dataY = trainY
        self.dims = dims
        self.num_batch = num_batch
        self.test_size = test_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.epoch = epoch
        self.build_dae()

    def build_dae(self):
        self.encoder, self.decoder, self.autoencoder = mnist_dae(self.dims)
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
            plt.imshow(xtest_n[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display Reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
    def predict(self,testX):
        if len(testX.shape) is not 2:
            testX = testX.reshape(testX.shape[0],-1)
        return self.autoencoder.predict(testX)
    ## dynamical system using the trained DAE model
    def apply_DS(self, testX, vr=0.9, max_iter=30):
        if len(testX.shape) is not 2:
            testX = testX.reshape(testX.shape[0],-1)
        revX = self.autoencoder.predict(testX)
        projX = vr*testX + (1-vr)*revX
        for i in range(max_iter):
            revX = self.autoencoder.predict(projX)
            projX = vr*projX +(1-vr)*revX

        return projX
    def save(self,save_path = '../results/mnist_dae.h5'):
        self.autoencoder.save(save_path)
    def load_model(self,load_path = '../results/mnist_dae.h5'):
        self.autoencoder = load_model(load_path)


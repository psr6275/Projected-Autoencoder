from keras.layers import Input, Dense,Dropout, Flatten, Lambda,Layer
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Conv2DTranspose
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
from mnist_ds import *
import matplotlib.pyplot as plt
from keras.models import load_model
#from mnist_dae import corrupt

def corrupt_toy(x,scale = 0.3,rep = 1,noise_type = "gaussian"):
    x_rep = np.repeat(x,rep,axis=0)
    if noise_type =='gaussian':
        noise = np.random.normal(0,scale = scale,size=x_rep.shape)
        x_crr = x_rep+noise
    elif noise_type == 'uniform':
        noise = np.random.uniform(-scale,scale,size = x_rep.shape)
        x_crr = x_rep+noise
    else:
        print("you sould select noise_type in [gaussian,uniform]")
    return x_rep, x_crr
def toy_dae(input_dim = (3,),dims = [100]):
    input_ex = Input(shape = input_dim)
    encoded = input_ex
    for idx, dim in enumerate(dims):
        encoded = Dense(dim,activation='relu',kernel_initializer = 'glorot_normal')(encoded)
    decoded = encoded
    for idx, dim in enumerate(dims[::-1]):
        if idx<len(dims)-1:
            decoded = Dense(dim,activation='relu',kernel_initializer='glorot_normal')(decoded)
        else:
            decoded = Dense(input_dim[0],activation='linear',kernel_initializer='glorot_normal')(decoded)

    encoder = Model(input_ex,encoded)
    autoencoder = Model(input_ex,decoded)
    input_z = Input(shape=(dims[-1],))
    decoder_layers = autoencoder.layers[len(dims)+1:]
    z_decoded = input_z
    for lyr in decoder_layers:
        z_decoded = lyr(z_decoded)
    decoder = Model(input_z,z_decoded)
    print(autoencoder.summary())
    return encoder, decoder, autoencoder

class Toy_DAE:
    def __init__(self,trainX,dims=[100],num_batch = 128,test_size=0.3,loss_type = "mean_squared_error",
                 noise_type = "gaussian",noise_scale=0.3,noise_rep = 3,epoch=30):
        self.data = trainX
        self.dims = dims
        self.num_batch = num_batch
        self.test_size = test_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_rep = noise_rep
        self.epoch = epoch
        self.loss_type = loss_type
        self.build_dae()

    def build_dae(self):
        self.input_dim = (self.data.shape[1],)
        self.encoder, self.decoder, self.autoencoder = toy_dae(self.input_dim,self.dims)

    def train_dae(self):
        self.autoencoder.compile(optimizer = 'nadam',loss = self.loss_type)
        self.trainX, self.trainXn = corrupt_toy(self.data,noise_type = self.noise_type,rep = self.noise_rep,scale = self.noise_scale)
        self.idxs = np.array(range(len(self.data)))
        val_num = int(len(self.idxs)*self.test_size)
        xtr_o = self.trainX[self.idxs[val_num:]]
        xtr_n = self.trainXn[self.idxs[val_num:]]
        xval_o = self.trainX[self.idxs[:val_num]]
        xval_n = self.trainXn[self.idxs[:val_num]]
        print("loss type is ", self.loss_type)
        print("train the DAE model with noise ", self.noise_type, " (",self.noise_scale,")")
        self.autoencoder.fit(xtr_n,xtr_o,epochs = self.epoch, batch_size = self.num_batch,
                             shuffle = True, validation_data = (xval_n, xval_o),
                             callbacks = [TensorBoard(log_dir = '../logs/toy',histogram_freq=0,
                                                      write_graph=False)])
    def predict(self,testx):
        return self.autoencoder.predict(testx)
    def apply_DS(self,testX, vr=0.9,max_iter = 30):
        revX = self.autoencoder.predict(testX)
        projX = vr*testX + (1-vr)*revX
        for i in range(max_iter):
            revX = self.autoencoder.predict(projX)
        return projX


    def save(self,save_path = '../results/toy_dae.h5'):
        self.autoencoder.save(save_path)
    def load_model(self,load_path = '../results/toy_dae.h5'):
        self.autoencoder = load_model(load_path)
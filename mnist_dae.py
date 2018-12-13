from keras.layers import Input, Dense,Dropout, Flatten, Lambda
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split as tts
from keras.utils import np_utils, plot_model
from keras.layers.convolutional import COnvoultion2D, MaxPooling2D, Conv2D
from keras import backend as K
from keras.losses import categorical_hinge, categorical_crossentropy

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
    input_img = Input(shape = (dimx[0],))
    encoded = input_img
    for idx, dim in enumerate(dims[1:]):
        encoded = Dense(dim, activation='elu')(encoded)

    decoded = encoded
    for idx, dim in enumerate(dims[:-1][::-1]):
        if idx < len(dims)-2:
            decoded = Dense(dim,activation='elu')(decoded)
        else:
            decoded = Dense(dim,activation='sigmoid')(decoded)
    encoder = Model(nput_img,encoded)
    autoencoder = Model(input_img,decoded)
    input_z = Input(shape=(dims[-1],))
    decoder_layers = autoencoder.layers[len(dims):]
    z_encoded = input_z
    for lyr in decoder_layers:
        z_decoded = lyr(z_decoded)
    decoder = Model(input_z,z_decoded)
    return encoder, decoder, autoencoder

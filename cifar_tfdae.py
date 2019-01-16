from keras.utils import np_utils, plot_model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from six.moves import xrange
import os,sys
from cifar_clf import Next_Batch
import matplotlib.pyplot as plt
from mnist_dae import corrupt

##Define DAE network! using TF
def cifar_cdae_over(x,train_mode = True):
    with tf.variable_scope("DAE",reuse=tf.AUTO_REUSE) as vs:
        conv1 = tf.layers.conv2d(inputs = x, filters = 64,
                                 kernel_size = [9,9], padding = 'same',
                                 activation = tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs = conv1, filters = 32,
                                 kernel_size = [1,1], padding = 'same',
                                 activation = tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs = conv2,filters = 3,
                                 kernel_size = [5,5],padding = 'same')
        out = tf.clip_by_value(conv3,0,1)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

##Define noised tensor depending on noise_type ( but mainly from gaussian noise)
def Corrupt_tensor(x,noise_type = "gaussian",noise_scale=0.1):
    if noise_type is "gaussian":
        noise = tf.random_normal(tf.shape(x),stddev=noise_scale)
    #elif noise_type is "uniform":
    else:
        noise = tf.random_uniform(tf.shape(x),maxval=noise_scale)
    #elif noise_type is "corruption":
    #    noise = tf.random.
    return tf.clip_by_value(x+noise,0,1)


class Cifar10_DAE:
    def __init__(self,trainX, trainY = None, num_batch = 128,test_size = 0.3,dae_type = "over",
                 loss_type = "mse",noise_type = "gaussian",noise_scale = 0.1,lr = 0.01,num_epoch=30,
                 optim_type = "adaDelta",split_num = None,log_path = '../logs/',
                 save_path = '../data/'):
        self.dataX = trainX
        self.num_data = len(trainX)
        if trainY is not None:
            self.dataY = trainY
        else:
            self.dataY = np.ones((self.num_data,))
        self.num_batch = num_batch
        self.test_size = test_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.num_epoch = num_epoch
        self.loss_type = loss_type
        self.dae_type = dae_type
        self.lr = lr
        if split_num is not None:
            #split_num is related to memory
            self.split_num = split_num
        else:
            self.split_num = 2*num_batch
        if optim_type not in ["adaDelta","rms","adam"]:
            print("Please, specify an appropriate optimizer among adaDelta, rms, adam.")
        else:
            self.optim_type = optim_type
        self.build_model()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.log_path = log_path
        self.save_path = save_path


    def build_model(self):
        self.input_imgs = tf.placeholder(tf.float32,shape=(None,32,32,3))
        self.noise_imgs = Corrupt_tensor(self.input_imgs,self.noise_type,self.noise_scale)
        self.eval_imgs = tf.placeholder(tf.float32,shape = (None,32,32,3))

        if self.dae_type == "over":
            self.recon,self.dae_variables = cifar_cdae_over(self.noise_imgs)
            self.recon_eval,_ = cifar_cdae_over(self.eval_imgs,train_mode = False)
        else:
            self.recon,self.dae_variables = cifar_cdae(self.noise_imgs)
            self.recon_eval,_ = cifar_cdae(self.evaL-imgs,train_mode = False)

        ##loss part
        if self.loss_type == "mse":
            self.loss = tf.reduce_mean(tf.square(self.input_imgs-self.recon))
            self.loss_eval = tf.reduce_mean(tf.square(self.eval_imgs-self.recon_eval))
        elif self.loss_type == "bce":
            self.loss = tf.keras.backend.binary_crossentropy(self.input_imgs,self.recon)
            self.loss_eval = tf.keras.backend.binary_crossentropy(self.eval_imgs,self.recon_eval)
        elif self.loss_type == "l1":
            self.loss = tf.reduce_mean(tf.abs(self.input_imgs-self.recon))
            self.loss_eval = tf.reduce_mean(tf.abs(self.eval_imgs-self.recon_eval))
        else:
            print("Error!: You should specify an appropriate loss type.")

        ##train variables
        with tf.variable_scope('TrainVar',reuse = tf.AUTO_REUSE):
            if self.optim_type == "rms":
                self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss,var_list = self.dae_variables)
            elif self.optim_type == "adam":
                self.train_step = tf.train.AdamOptimizer().minimize(self.loss,var_list = self.dae_variables)
            elif self.optim_type == "adaDelta":
                self.train_step = tf.train.AdadeltaOptimizer().minimize(self.loss,var_list = self.dae_variables)


    def train(self,ckpt_name = "cifar10_dae.ckpt",num_viz = False):
        Xtr, Xval, Ytr, Yval = tts(self.dataX,self.dataY,test_size = self.test_size)
        self.sess.run(tf.global_variables_initializer())
        self.batch_gen = Next_Batch(len(Ytr),self.num_batch)
        num_iter = self.num_epoch*self.batch_gen.batch_len
        for i in range(num_iter):
            batch_idxs = self.batch_gen.get_batch()
            X_batch = Xtr[batch_idxs]
            self.sess.run(self.train_step,feed_dict={self.input_imgs:X_batch})

            if i %500 ==0:
                train_loss, train_loss_cln = self.loss_score(Xtr)
                val_loss, val_loss_cln = self.loss_score(Xval)

                print("[{}/{}] (Loss) Train_n: {:.6f} Train_cln: {:.6f} Val_n: {:.6f} Val_cln: {:.6f}".\
                      format(i,num_iter,train_loss,train_loss_cln, val_loss, val_loss_cln))
                if num_viz:
                    nviz_imgs,rnviz_imgs,rcviz_imgs = self.sess.run([self.noise_imgs,self.recon,self.recon_eval],feed_dict={self.input_imgs:Xval[:num_viz],\
                            self.eval_imgs:Xval[:num_viz]})
                    for vi in range(num_viz):
                        ax = plt.subplot(num_viz,4,4*vi+1)
                        plt.imshow(Xval[vi])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax = plt.subplot(num_viz,4,4*vi+2)
                        plt.imshow(nviz_imgs[vi])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax = plt.subplot(num_viz,4,4*vi+3)
                        plt.imshow(rcviz_imgs[vi])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax = plt.subplot(num_viz,4,4*vi+4)
                        plt.imshow(rnviz_imgs[vi])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                    plt.show()


        self.save_path = self.saver.save(self.sess,self.log_path+ckpt_name)
        print("saved path: ",self.save_path)
    def loss_score(self,input_tensor):
        data_num = len(input_tensor)

        if data_num<self.split_num:
            loss_n,loss_cln = self.sess.run([self.loss,self.loss_eval],
                                            feed_dict = {self.input_imgs:input_tensor,self.eval_imgs:input_tensor})
        else:
            batch_num = int(data_num/self.split_num)
            loss_cln = 0
            loss_n = 0
            for i in range(batch_num):
                batchX = input_tensor[i*self.split_num:(i+1)*self.split_num]
                loss1, loss2 = self.sess.run([self.loss,self.loss_eval],
                                             feed_dict = {self.input_imgs:batchX, self.eval_imgs:batchX})
                loss_n += loss1
                loss_cln += loss2
            loss_n /=batch_num
            loss_cln /= batch_num

        return loss_n,loss_cln
    def predict(self,testX):
        data_num = len(testX)

        if data_num < self.split_num:
            recon = self.sess.run(self.recon_eval,feed_dict = {self.eval_imgs:testX})
        else:
            batch_num = np.ceil(data_num/self.split_num)
            recon = np.zeros(testX.shape)
            for i in range(batch_num):
                if i <batch_num-1:
                    recon[i*self.split_num:(i+1)*self.split_num] = \
                            self.sess.run(self.recon_eval,feed_dict = {self.eval_imgs:\
                                                                        testX[i*self.split_num:(i+1)*self.split_num]})
                else:
                    recon[i*self.split_num:] = \
                            self.sess.run(self.recon_eval, feed_dict = {self.eval_eimgs:testX[i*self.split_num:]})

        return recon
    def restore(self,ckpt_path = None):
        if ckpt_path is None:
            print("Please, specify the path of checkpoint")
        else:
            self.saver.restore(self.sess,ckpt_path)
            print("Success!")
    def apply_DS(self,testX, vr=0.9,max_iter = 30):
        revX = self.predict(testX)
        projX = vr*testX+(1-vr)*revX
        for i in range(max_iter):
            revX = self.predict(projX)
            projX = vr*projX + (1-vr)*revX

        return projX

    def plot_imgs(self,testX, noise_type = 'peppSalt',noise_scale = 0.3):
        xtest_o,xtest_n = corrupt(testX,scale=noise_scale,noise_type = noise_type)
        decoded_imgs = self.predict(xtest_n)
        n = 10
        plt.figure(figsize = (20,4))
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            plt.imshow(xtest_n[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2,n,i+1+n)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

import os
import scipy.misc
import numpy as np
import pickle
import tensorflow as tf
from collections import deque
from itertools import chain
from glob import glob
from tqdm import trange
from io import StringIO

from models import *

def corrupt(x,noise_type='gaussian',noise_param=0.3):
    """
    Take an input tensor and add corruption, Gaussian or pepper and salt noises
    
    Parameters
    ----------
    x: Tensor/Placeholder
        Input to corrupt
    noise_type: String in {'gaussian','peppSalt','corruption'}
    noise_param: Float {sigma, degree, ratio}

    Returns
    -------
    x_corrupted: Tensor
    """

    if noise_type == 'corrupt':
        mask_dist = dist.Bernoulli(probs=1-noise_param, detype = tf.float32)
        mask = mask_dist.sample(sample_shape=tf.shape(x))
        corrupted = tf.multiply(x,mask)
    elif noise_type == "peppSalt":
        noise_dist = dist.Categorical(probs = [noise_param/2,1-noise_param, noise_param/2])
        noise = tf.cast(noise_dist.sample(sample_shape = tf.shape(x))-1,dtype = tf.float32)
        corrupted = tf.clip_by_value(tf.add(x,noise),0.0,1.0)
    else:
        noise = tf.random_normal(shape = tf.shape(x), mean = 0.0,
                stddev = noise_param)
        corrupted = tf.clip_by_value(tf.add(x,noise),0.0,1.0)

    return corrupted

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def nchw_to_nhwc(x):
    return tf.transpose(x,[0,2,3,1])

def norm_img(image,data_format = None):
    image = image/255.0
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm,data_format):
    return tf.clip_by_value(to_nhwc(norm*255.0, data_format),0,255)
class Trainer(object):
    def __init__(self,config,data_loader=None):
        self.config = config
        self.data_loader = data_loader
        self.network = config.network_type # kind of network like dense or conv
        #self.activation_fn = config.activation_fn
        if config.activation_fn =="relu":
            self.activation_fn = tf.nn.relu
        elif config.activation_fn =="elu":
            self.activation_fn = tf.nn.elu
        else:
            raise Exception("[!] Cuation! You should select the one among relu and elu")
        self.layers = [int(aa) for aa in config.layers.split(" ")]
        self.noise = config.noise #kind of noises like gaussian, corruption
        self.noise_param = config.noise_param

        self.data_format = config.data_format # Specify the shape of image data

        self.lr = tf.Variable(config.lr,name='lr')
        self.lr_update = tf.assign(self.lr,tf.maximum(self.lr*0.5,config.lr_lower_boundary),name='lr_update')
        self.optimizer = config.optimizer # Specify the optimization algorithm like adam!
        self.cost_function = config.cost_function # among sq_loss, abs_loss, binary_cross_entropy

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        
        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.step = tf.Variable(0,name = 'step', trainable=False)

        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir = self.model_dir,
                                is_chief = True, saver = self.saver,
                                summary_op = None,
                                summary_writer = self.summary_writer,
                                save_model_secs = 10,
                                global_step = self.step, 
                                ready_for_local_init_op = None)

        gpu_options = tf.GPUOptions(allow_growth = True)
        sess_config = tf.ConfigProto(allow_soft_placement = True,
                                    gpu_options = gpu_options)
        if self.is_train:
            self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        else:
            self.sess = tf.Session()
            #dirty way to bypass graph finalization error
            g = tf.get_default_graph()
            g._finalized = False
            try:
                self.saver.restore(self.sess,tf.train.latest_checkpoint(self.model_dir))
            except:
                ckpt_path = tf.train.get_checkpoint_state(self.model_dir)
                self.saver.restore(self.sess, ckpt_path.all_model_checkpoint_oaths[-2])

    def train(self):
        
        self.start_step = self.sess.run(self.step)
        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step,self.lr_update_step)

        for step in trange(self.start_step,self.max_step):
            fetch_dict = {
                    "optim":self.optim
                    }
            if step %self.log_step ==0:
                fetch_dict.update({
                    "summary":self.summary_op,
                    "loss":self.loss,
                    "lr":self.lr
                    })
            result = self.sess.run(fetch_dict)
            
            
            if step % self.log_step ==0:
                self.summary_writer.add_summary(result['summary'],step)
                self.summary_writer.flush()

                loss = result['loss']
                print("[{}/{}] Loss: {:.6f}".format(step,self.max_step,loss))

            if step % self.lr_update_step == self.lr_update_step -1:
                self.sess.run(self.lr_update)
        self.ckpt_path = self.saver.save(self.sess,self.model_dir,self.step)
                

    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x,self.data_format)
        crrX = corrupt(x,self.noise,self.noise_param)
        self.crrX = denorm_img(crrX,self.data_format)
        temp_num = x.get_shape().as_list()

        if self.network is "dense":
            x = tf.reshape(x,[-1,self.layers[0]])
            rx, z, variables = AE_dense(crrX,self.layers,self.activation_fn)
        elif self.network is "conv":
            rx, z, variables = AE_conv(crrX,self.layers,self.activation_fn,self.data_format)
        else: # need to add another network like convNet
            raise Exception("You should add other network like convNet in models.py")
        self.rx = denorm_img(rx,self.data_format)
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            raise Exception("You should add another optimizer in trainer.py ")
        
        if self.cost_function == 'abs_loss':
            self.loss = tf.reduce_mean(tf.abs(rx-x))
        elif self.cost_function =='sq_loss':
            self.loss = tf.reduce_mean(tf.square(rx-x))
        else:
            # x and rx should be in the range of 0~1
            self.loss = -tf.reduce_mean(x*tf.log(rx+1e-10)+(1-x)*tf.log(1-rx+1e-10))
       
        self.optim = optimizer.minimize(self.loss,global_step = self.step,var_list = variables)
       
       
        self.summary_op = tf.summary.merge([
            #tf.summary.image("rx",self.rx),
            #tf.summary.image("x",self.x),
            #tf.summary.image("crrX",self.crrX),

            tf.summary.scalar("loss",self.loss),
            tf.summary.scalar("lr",self.lr)
            ])


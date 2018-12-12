from keras.utils import np_utils, plot_model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts

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

def cnn_model_explicit(x,train_mode = True):
        
    with tf.variable_scope("CLF",reuse = tf.AUTO_REUSE) as vs:
        conv1 = tf.layers.conv2d(inputs = x, filters=32,
                    kernel_size=[5,5], padding='same',
                    activation= tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = [2,2],
                    strides = 2)
        conv2 = tf.layers.conv2d(inputs = pool1,
                    filters = 64, kernel_size = [5,5],
                    padding = 'same', activation = tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2],
                    strides = 2)
        pool2_flat = tf.layers.flatten(pool2)
        dense = tf.layers.dense(inputs = pool2_flat,units=1024, 
                    activation = tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, 
                    training =train_mode)
        logits = tf.layers.dense(inputs=dropout, units = 10)
    variables = tf.contrib.framework.get_variables(vs)
    return logits, variables


class Next_Batch:
    def __init__(self, num_data, num_batch, idxs=None):
        self.num_data = num_data
        self.num_batch = num_batch
        self.idxs = None
        self.batch_len = int(num_data / num_batch)
        self.idxs = np.arange(0, num_data)
        np.random.shuffle(self.idxs)
        self.elapsed_batch = 0
        self.num_epoch = 0

    def shuffle_idxs(self):
        np.random.shuffle(self.idxs)

    def get_batch(self):
        if self.elapsed_batch > self.batch_len - 1:
            self.shuffle_idxs()
            self.elapsed_batch = 0
            self.num_epoch += 1
        self.elapsed_batch += 1
        return self.idxs[(self.elapsed_batch - 1) * self.num_batch:self.elapsed_batch * self.num_batch]


class Mnist_CNN:
    def __init__(self, trainX, trainY, num_batch=128, num_epoch=10, lr=1e-3, val_ratio=0.3):
        self.trainX = trainX
        self.trainY = trainY
        self.val_ratio = val_ratio
        self.num_data = len(trainY)
        self.num_batch = num_batch
        self.num_epoch = num_epoch
        self.lr = lr
        self.build_model()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def build_model(self):
        self.input_imgs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.labels = tf.placeholder(tf.int32, shape=(None, 10))
        self.eval_imgs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.labels_eval = tf.placeholder(tf.int32, shape=(None, 10))
        self.logits, self.clf_variables = cnn_model_explicit(self.input_imgs)
        self.y_pred = tf.nn.softmax(self.logits, name="prediction")
        self.logit_eval, _ = cnn_model_explicit(self.eval_imgs, train_mode=False)
        self.y_eval = tf.nn.softmax(self.logit_eval, name="evaluation")
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        # self.loss_eval = tf.reduce_mean()
        with tf.variable_scope('TrainVal', reuse=tf.AUTO_REUSE):
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=self.clf_variables)
            correct_prediction = tf.equal(tf.argmax(self.y_eval, 1), tf.argmax(self.labels_eval, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, save_path="../logs/mnist_CNN_clf.ckpt"):
        Xtr, Xval, Ytr, Yval = tts(self.trainX, self.trainY, test_size=self.val_ratio)
        self.sess.run(tf.global_variables_initializer())
        self.batch_gen = Next_Batch(int(self.num_data * self.val_ratio), self.num_batch)
        num_iter = self.num_epoch * self.batch_gen.batch_len
        for i in range(num_iter):
            batch_idxs = self.batch_gen.get_batch()
            X_batch = Xtr[batch_idxs]
            Y_batch = Ytr[batch_idxs]
            self.sess.run(self.train_step, feed_dict={self.input_imgs: X_batch,
                                                      self.labels: Y_batch})
            if i % 500 == 0:
                train_loss, train_acc = self.sess.run([self.loss, self.accuracy], feed_dict= \
                    {self.input_imgs: X_batch, self.labels: Y_batch,
                     self.eval_imgs: Xtr, self.labels_eval: Ytr})
                val_acc = self.sess.run(self.accuracy, feed_dict={self.eval_imgs: Xval, self.labels_eval: Yval})
                print("[{}/{}] Loss: {:.6f} Train_Acc: {:.4f} Val_Acc: {:.4f}". \
                      format(i, num_iter, train_loss, train_acc, val_acc))
        self.save_path = self.saver.save(self.sess, save_path)
        print("saved path: ", self.save_path)

    def predict(self, testX):
        y_pred = self.sess.run(self.y_eval, feed_dict={self.eval_imgs: testX})
        return y_pred

    def accuracy_score(self, testX, testY):
        acc = self.sess.run(self.accuracy, feed_dict={self.eval_imgs: testX, self.labels_eval: testY})
        return acc

    def restore(self, ckpt_path=None):
        if ckpt_path is None:
            print('Please, specify the path for checkpoint')
        else:
            self.saver.restore(self.sess, ckpt_path)
            print("Success!")
    def targeted_attack(self,testX,testY,alpha = 0.1,max_iter=20,eps=32./255.,save_path = None):
        """

        :param testX: we want to bother
        :param testY:
        :param alpha:
        :param max_iter:
        :param eps:
        :param save_path: should contain the last '/' for the next file name!
        :return:
        """
        adv_imgs = tf.placeholder(tf.float32,shape=(None,28,28,1))
        self.labels_adv = tf.placeholder(tf.int32,shape=(10,))
        logit_test,_ = cnn_model_explicit(adv_imgs,train_mode = False)
        adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit_test,
                labels = self.labels_adv)
        grad = tf.gradients(adv_loss,adv_imgs)

        self.advX = np.copy(testX).astype('float32')
        yte = np.argmax(Y_test,axis=1)
        yte_adv = (yte+np.random.randint(low=1,hugh=9,size=(Y_test.shape[0],)))%10
        self.advY = np_utils.to_categorical(yte_adv)
        adv_list = []

        for i in range(len(Y_test)):
            tmp = np.expand_dims(advX[i],0)
            lower = np.clip(tmp-eps,0,1)
            upper = np.clip(tmp+eps,0,1)

            for itr in range(max_iter):
                g = self.sess.run(grad,{adv_imgs:tmp,self.labelse_adv:self.advY})
                tmp = tmp - alpha*np.sign(g[0])
                tmp = np.clip(tmp,lower,upper)
            adv_list.append(list(tmp[0]))

        self.advX = np.array(adv_list)
        if save_path is not None:
            np.save(save_path+'mnist_advX_%d_%d_%d'%(i,yte,yte_adv[i]))

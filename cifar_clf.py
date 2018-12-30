from keras.utils import np_utils, plot_model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from six.moves import xrange
import os,sys
# to implement cifar10 we need some distortions of inputs
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)








def cnn_model_explicit(x, train_mode=True):
    with tf.variable_scope("CLF", reuse=tf.AUTO_REUSE) as vs:
        conv1 = tf.layers.conv2d(inputs=x, filters=64,
                                 kernel_size=[5, 5], padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                        strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=50, kernel_size=[5, 5],
                                 padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                        strides=2)
        pool2_flat = tf.layers.flatten(pool2)
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1050,
                                activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4,
                                    training=train_mode)
        dense2 = tf.layers.dense(inputs=dropout1, units=100,
                                activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4,
                                    training=train_mode)
        logits = tf.layers.dense(inputs=dropout2, units=10)
    variables = tf.contrib.framework.get_variables(vs)
    return logits, variables


class Next_Batch:
    def __init__(self, num_data, num_batch, idxs=None):
        self.num_data = num_data
        self.num_batch = num_batch
        self.idxs = idxs
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


class Cifar10_CNN:
    def __init__(self, trainX, trainY, num_batch=128, num_epoch=10, lr=1e-3, val_ratio=0.3,
                 log_path='../logs/', save_path='../data/'):
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
        # if log_path is not None:
        self.log_path = log_path
        # if save_path is not None:
        self.save_path = save_path

    def build_model(self):
        self.input_imgs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.labels = tf.placeholder(tf.int32, shape=(None, 10))
        self.eval_imgs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
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

    def train(self, ckpt_name="cifar10_CNN_clf.ckpt"):
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
                     self.eval_imgs: X_batch, self.labels_eval: Y_batch})
                val_acc = 0
                cv_idxs = np.array(range(len(batch_idxs)))
                for cv in range(10):
                    np.shuffle(cv_idxs)
                    val_acc = val_acc + self.sess.run(self.accuracy, feed_dict={self.eval_imgs: Xval[cv_idxs[:self.num_batch]], 
                        self.labels_eval: Yval[cv_idxs[:self.num_batch]]})
                val_acc = val_acc/10
                print("[{}/{}] Loss: {:.6f} Train_Acc: {:.4f} Val_Acc: {:.4f}". \
                      format(i, num_iter, train_loss, train_acc, val_acc))
        self.save_path = self.saver.save(self.sess, self.log_path + ckpt_name)
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

    def targeted_attack(self, testX, testY, alpha=0.1, max_iter=20, eps=16. / 255., file_name=None):
        """

        :param testX: we want to bother
        :param testY:
        :param alpha:
        :param max_iter:
        :param eps:
        :param save_path: should contain the last '/' for the next file name!
        :return:
        """
        adv_imgs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.labels_adv = tf.placeholder(tf.int32, shape=(10,))
        logit_test, _ = cnn_model_explicit(adv_imgs, train_mode=False)
        adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit_test,
                                                           labels=self.labels_adv)
        grad = tf.gradients(adv_loss, adv_imgs)

        self.advX = np.copy(testX).astype('float32')
        yte = np.argmax(testY, axis=1)
        yte_adv = (yte + np.random.randint(low=1, high=9, size=(testY.shape[0],))) % 10
        self.advY = np_utils.to_categorical(yte_adv)
        adv_list = []

        for i in range(len(testY)):
            tmp = np.expand_dims(self.advX[i], 0)
            lower = np.clip(tmp - eps, 0, 1)
            upper = np.clip(tmp + eps, 0, 1)

            for itr in range(max_iter):
                g = self.sess.run(grad, {adv_imgs: tmp, self.labels_adv: self.advY[i]})
                tmp = tmp - alpha * np.sign(g[0])
                tmp = np.clip(tmp, lower, upper)
            adv_list.append(list(tmp[0]))
            if i % 100 == 0:
                print("[{}/{}] processed".format(i, len(testY)))

        self.advX = np.array(adv_list)
        if file_name is not None:
            np.save(self.save_path + file_name, self.advX)

    def save_np(self, np_array, file_name):
        np.save(self.save_path + file_name, np_array)
        print("Save: ", self.save_path + file_name)

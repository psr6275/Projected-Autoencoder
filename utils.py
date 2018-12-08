from __future__ import print_function

import os
import math
import json
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image


def batch_generator(X, batch_size, data_format,y=None,  split = None, seed=None):
    """
    Generate batch data from Tensors of X and y
    """
    if y:
        datasets = [X,y]
    else:
        datasets = [X]
    
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3* batch_size

    if y:
        image_batch, label_batch= tf.train.shuffle_batch(datasets,
                                    batch_size = batch_size,
                                    enqueue_many=True,
                                    num_threads=4,capacity=capacity,
                                    min_after_dequeue = min_after_dequeue,
                                    name = 'training_data')
        return tf.to_float(image_batch), tf.to_float(label_batch)
    else:
        image_batch = tf.train.shuffle_batch(datasets,
                                    batch_size = batch_size,
                                    enqueue_many=True,
                                    num_threads=4,capacity=capacity,
                                    min_after_dequeue = min_after_dequeue,
                                    name = 'training_data')

        return tf.to_float(image_batch)


def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split)
    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))
        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
                [image], batch_size=batch_size,
                num_threads=4, capacity=capacity,
                min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startwith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startwith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset,
                                    config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset,get_time())

        if not hasattr(config,'model_dir'):
            config.model_dir = os.path.join(config.log_dir, config.model_name)
        config.data_path = os.path.join(config.data_dir,config.dataset)

        for path in [config.log_dir, config.data_dir, config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path,'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys = True)

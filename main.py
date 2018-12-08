from utils import prepare_dirs_and_logger,get_loader,batch_generator
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import save_config

def main(config):
    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle=True
    else:
        setattr(config,'batch_size',64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle=False
    if config.dataset == 'mnist' :
        mnist = tf.keras.datasets.mnist
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        data_loader = batch_generator(x_train, config.batch_size, config.data_format)
    elif config.dataset == 'cifar10':
        cifar = tf.keras.datasets.cifar10
        (x_train,y_train),(x_test,y_test) = cifar.load_data()
        data_loader = batch_generator(x_train, config.batch_size, config.data_format)
    else:
        data_loader = get_loader(
            data_path, config.batch_size,config.scale_size, config.data_format)
    trainer = Trainer(config,data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify load_path to load a pretrained model")
        trainer.test()

if __name__ =="__main__":
    config, unparsed = get_config()
    main(config)


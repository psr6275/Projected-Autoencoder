import argparse

def str2bool(v):
    return v.lower() in ('true','1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type',type=str,default='dense',choices=['dense','conv'])
net_arg.add_argument('--activation_fn',type=str,default='elu',choices = ['relu','elu'])
net_arg.add_argument('--layers',type=str, default='3 64 128 32 1024',
            help='network structure with space!')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset',type=str,default='mnist',choices = ['mnist','cifar10'])
data_arg.add_argument('--split',type=str,default='train')
data_arg.add_argument('--batch_size',type=int,default=128)
data_arg.add_argument('--num_worker',type=int, default=16)
data_arg.add_argument('--noise',type=str,default='gaussian',choices=['gaussian','saltPepp','corruption'])
data_arg.add_argument('--noise_param',type=float,default=0.3)
data_arg.add_argument('--scale_size',type=int,default=64)

# Training / Test Parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train',type=str2bool, default=True)
train_arg.add_argument('--optimizer',type=str,default='adam')
train_arg.add_argument('--max_step',type=int,default=100000)
train_arg.add_argument('--lr_update_step',type=int,default=15000,choices=[5000,7500,15000])
train_arg.add_argument('--lr',type=float,default=0.001)
train_arg.add_argument('--lr_lower_boundary',type=float,default=0.0001)
train_arg.add_argument('--use_gpu',type=str2bool,default=True)
train_arg.add_argument('--save_trainer',type=str2bool,default=False)
train_arg.add_argument('--cost_function',type=str,default='binary_cross_entropy',
        choices=['binary_cross_entropy','sq_loss','abs_loss'])

#Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path',type=str,default='')
misc_arg.add_argument('--log_step',type=int,default=50)
misc_arg.add_argument('--save_step',type=int,default=5000)
misc_arg.add_argument('--log_dir',type=str,default='logs')
misc_arg.add_argument('--data_dir',type=str,default='data')
misc_arg.add_argument('--test_data_path',type=str,default=None,
        help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--random_seed',type=int,default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'

    if config.dataset == 'mnist':
        data_format = 'gray'

    setattr(config, 'data_format', data_format)
    print(config.dataset)
    return config, unparsed



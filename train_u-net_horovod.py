from __future__ import print_function

import tensorflow as tf
import read_sunrgbd_data
from PIL import Image
import argparse

from UNet import unet
import time
import numpy as np

import horovod.tensorflow as hvd

headless = 'True'
img_width  = 320
img_height = 240

# Training settings
parser = argparse.ArgumentParser(description='plotting example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
                    
args = parser.parse_args()

rows = np.int(np.ceil(np.sqrt(args.batch_size)))
cols = np.int(np.ceil(args.batch_size / rows))

hvd.init()


# SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD",
#                                             "/se3netsproject/data/multijtdata/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/motion0",
#                                             img_type='depth')

img_type = 'rgb'

SUNRGBD_dataset = read_sunrgbd_data.dataset("SceneNetRGBD",
                                            "/se3netsproject/train_img_label_gt3_scenenet_dataset.txt",
                                            img_type=img_type)

max_labels = 14
batch_size = 30
learning_rate = 1e-3
iter_num = 0

logs_path = '/tensorboard/tf-summary-logs/'
checkpoint_dir = '/tensorboard/checkpoints' if hvd.rank() == 0 else None

global_step = tf.train.get_or_create_global_step()

UNET = unet(batch_size, img_height, img_width, learning_rate, sess=None, num_classes=max_labels, is_training=True,
            img_type=img_type, use_horovod=True, global_step=global_step)

hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=600000)# // hvd.size())
    ]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

summary_writers = []
write_images_per_sec_files = False

num_epochs = 1
base_lr = 0.1
cur_learning_rate = base_lr
iters_per_epoch = int(SUNRGBD_dataset.dataset_size / ( batch_size * hvd.size()))

with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as mon_sess:

    for i in range(0, hvd.size()):
        summary_writer = tf.summary.FileWriter(logs_path + 'plot_{:03d}'.format(hvd.rank()),
                                               graph=tf.get_default_graph())
        summary_writers.append(summary_writer)

    UNET.add_session(mon_sess)

    while not mon_sess.should_stop():

        # Run a training step synchronously.
        # Numba JIT speed up https://rushter.com/blog/numba-cython-python-optimization/
        # http://numba.pydata.org/numba-doc/dev/index.html
        img, label = SUNRGBD_dataset.get_random_shuffle(batch_size)
        batch_labels = label

        label = np.reshape(label, [-1])

        #TODO: add cosine learning rate scheduler
        #http://pytorch.org/docs/0.3.1/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR

        # if iter_num <= 100:
        #     UNET.set_learning_rate(learning_rate=1e-2 * hvd.size())

        # elif (iter_num > 100 and iter_num <= 3000):
        #     UNET.set_learning_rate(learning_rate=1e-3)# * hvd.size())
        # else:
        #     UNET.set_learning_rate(learning_rate=1e-4) #* hvd.size())

        if iter_num % iters_per_epoch == 0 and iter_num > 0:
            num_epochs = num_epochs + 1
            decay = np.floor((num_epochs-1)/5)
            cur_learning_rate = base_lr * np.power(0.95, decay)

        UNET.set_learning_rate(learning_rate=cur_learning_rate)


        #TODO: Implement Focal Loss https://arxiv.org/pdf/1708.02002.pdf
        #https://github.com/Kongsea/tensorflow/blob/fcf0063ec7d468237b8bca4814ef06e6350c8b1e/tensorflow/contrib/losses/python/losses/loss_ops.py

        batch_start = time.time()
        train_op, cost, pred, summary = UNET.train_batch(img, label)
        time_taken = time.time() - batch_start
        images_per_sec = batch_size / time_taken

        summary_writers[hvd.rank()].add_summary(summary, iter_num)
        summary_writers[hvd.rank()].flush()

        print('iter = ', iter_num, 'hvd_rank = ', hvd.rank(), 'cost = ', cost, 'images/sec = ', images_per_sec, 'batch_size = ', batch_size,
              'lr = ', cur_learning_rate, 'epochs = ', num_epochs, 'dataset_size = ', SUNRGBD_dataset.dataset_size, 'hvd_size =', hvd.size(),
              'iters_per_epoch = ', iters_per_epoch)

        if write_images_per_sec_files:
            fileName = logs_path + 'time_gpus_{:03d}_gpuid_{:03d}_iter_{:03d}.txt'.format(hvd.size(), hvd.rank(), iter_num)

            with open(fileName,'w') as f:
                f.write(str(images_per_sec))

        iter_num = iter_num + 1

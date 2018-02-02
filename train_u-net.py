from __future__ import print_function

import tensorflow as tf
import custom_layers_unet
import read_sunrgbd_data
from PIL import Image
import argparse

from UNet import unet
import time
import numpy as np


headless = 'True'
img_width  = 320
img_height = 240

if headless == False:

    import matplotlib.pyplot as pl
    import matplotlib as mpl

    pl.close('all')

    def tile_images(img, batch_size, rows, cols, rgb):

        batchImages = np.random.random((img_height*rows,img_width*cols,rgb))

        if rgb>1:
            batchImages = np.random.random((img_height*rows,img_width*cols,rgb))
        else:
            batchImages = np.random.random((img_height*rows,img_width*cols))

        for i in range(rows):
            for j in range(cols):
                if i*cols+j < batch_size:
                    if rgb > 1:
                        batchImages[0+i*img_height:(i+1)*img_height,0+j*img_width:(j+1)*img_width,:] = img[i*cols+j]
                    else:
                        batchImages[0+i*img_height:(i+1)*img_height,0+j*img_width:(j+1)*img_width]   = img[i*cols+j]

        return batchImages



# Training settings
parser = argparse.ArgumentParser(description='plotting example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
                    
args = parser.parse_args()

rows = np.int(np.ceil(np.sqrt(args.batch_size)))
cols = np.int(np.ceil(args.batch_size / rows))


#SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD","/data/ahanda/sunrgbd-meta-data/sunrgbd_rgb_training.txt")
#SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD","/data/workspace/sunrgbd-meta-data/sunrgbd_rgb_training.txt")
SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD",
                                            # "/data/ahanda/code/baxter_data_renderer/data/multijtdata/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/motion0",
                                            "/se3netsproject/data/multijtdata/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/motion0",
                                            img_type='depth')

# SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD","/Users/ankurhanda/workspace/code/sunrgbd-meta-data/sunrgbd_training.txt")

max_labels = 23

#inspired by http://jdherman.github.io/colormap/
# colour_code = [(0, 0, 0),(0,0,1),(0.9137,0.3490,0.1882), (0, 0.8549, 0),
#                (0.5843,0,0.9412),(0.8706,0.9451,0.0941),(1.0000,0.8078,0.8078),
#                (0,0.8784,0.8980),(0.4157,0.5333,0.8000),(0.4588,0.1137,0.1608),
#                (0.9412,0.1373,0.9216),(0,0.6549,0.6118),(0.9765,0.5451,0),
#                (0.8824,0.8980,0.7608)]

if headless == 'False':

    colour_code = np.random.rand(max_labels, 3)
    colour_code[0] = [0, 0, 0]

    cm = mpl.colors.ListedColormap(colour_code)

    fig, ax = pl.subplots()

    someImage = np.random.random((img_height*np.int(rows),img_width*np.int(cols),max_labels))
    some_img_argmax = np.argmax(someImage, axis=2)

    # Turn off axes and set axes limits
    im = ax.imshow(some_img_argmax, interpolation='none', cmap=cm)
    ax.axis('tight')
    ax.axis('off')

    # Set whitespace to 0
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.show()


config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.per_process_gpu_memory_fraction = 0.5


batch_size = 10
learning_rate = 1e-3
iter = 0

logs_path = '/tensorboard/tf-summary-logs/'
img_type = 'depth'

with tf.Session(config=config) as sess:

    UNET = unet(batch_size, img_height, img_width, learning_rate, sess, num_classes=max_labels, is_training=True,
                    img_type=img_type, use_horovod=True)

    print(str(hvd.local_rank()))

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    while True:

        img, label = SUNRGBD_dataset.get_random_shuffle(batch_size)
        batch_labels = label

        label = np.reshape(label, [-1])

        if iter <= 10:
            UNET.set_learning_rate(learning_rate=1e-2)

        elif (iter > 10 and iter <= 500):
            UNET.set_learning_rate(learning_rate=1e-3)
        else:
            UNET.set_learning_rate(learning_rate=1e-4)

        batch_start = time.time()
        train_op, cost, pred, summary = UNET.train_batch(img, label)
        time_taken = time.time() - batch_start
        images_per_sec = batch_size / time_taken

        summary_writer.add_summary(summary, iter)

        if headless == 'False':

            fig.show()
            pl.pause(0.00001)

            pred_class = np.argmax(pred, axis=3)
            batch_labels[batch_labels > 0] = 1

            pred_class_gt_mask = np.multiply(pred_class, batch_labels)


            batchImage = tile_images(pred_class_gt_mask, batch_size, rows, cols, 1)
            im.set_data(np.uint8(batchImage))

        print('iter = ', iter, 'max = ', batchImage.max(),'min = ', batchImage.min(), 'cost = ', cost, 'imaged = ', images_per_sec)

        iter = iter + 1
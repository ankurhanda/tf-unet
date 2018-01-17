from __future__ import print_function

import tensorflow as tf
import custom_layers_unet
import read_sunrgbd_data
from PIL import Image
import argparse

from UNet import UNet

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

pl.close('all')


def tile_images(img, batch_size, rows, cols, rgb):
    
    batchImages = np.random.random((240*rows,320*cols,rgb))

    if rgb>1:
        batchImages = np.random.random((240*rows,320*cols,rgb))
    else:
        batchImages = np.random.random((240*rows,320*cols))
        
    for i in range(rows):
        for j in range(cols):
            if i*cols+j < batch_size:
                if rgb>1:
                    batchImages[0+i*240:(i+1)*240,0+j*320:(j+1)*320,:] = img[i*cols+j]
                else:
                    batchImages[0+i*240:(i+1)*240,0+j*320:(j+1)*320]   = img[i*cols+j]
           
    return batchImages
        


# SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD","/media/ankur/nnseg/sunrgbd_training.txt")

# Parameters
#learning_rate = 0.1
training_iters = 200000
display_step = 10

base_learning_rate=0.1

#learning_rate = tf.placeholder(tf.float32, shape=[])

# tf Graph input
# x = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])
# y = tf.placeholder(tf.int32, batch_size*img_width*img_height)
# y_bool = tf.placeholder(tf.int32, batch_size*img_width*img_height)




# Training settings
parser = argparse.ArgumentParser(description='plotting example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
                    
args = parser.parse_args()

rows = np.int(np.ceil(np.sqrt(args.batch_size)))
cols = np.int(np.ceil(args.batch_size / rows))

#inspired by http://jdherman.github.io/colormap/
colour_code = [(0, 0, 0),(0,0,1),(0.9137,0.3490,0.1882), (0, 0.8549, 0),
               (0.5843,0,0.9412),(0.8706,0.9451,0.0941),(1.0000,0.8078,0.8078),
               (0,0.8784,0.8980),(0.4157,0.5333,0.8000),(0.4588,0.1137,0.1608),
               (0.9412,0.1373,0.9216),(0,0.6549,0.6118),(0.9765,0.5451,0),
               (0.8824,0.8980,0.7608)]
               
cm = mpl.colors.ListedColormap(colour_code)


fig, ax = pl.subplots()

someImage = np.random.random((240*np.int(rows),320*np.int(cols),14))
some_img_argmax = np.argmax(someImage,axis=2)
# Turn off axes and set axes limits
#im = ax.imshow(np.ones((240*rows,320*cols,3)), interpolation='none', cmap=cm)
im = ax.imshow(some_img_argmax, interpolation='none', cmap=cm)

ax.axis('tight')
ax.axis('off')

# Set whitespace to 0
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
fig.show()
class_weights = [0, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1]
weight_map = tf.constant(np.array(class_weights, dtype=np.float32))

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
init = tf.global_variables_initializer()


img_width  = 320
img_height = 240

batch_size = 20
learning_rate = 1e-3

with tf.Session(config=config) as sess:
    sess.run(init)

    unet = UNet(batch_size, img_height, img_width, learning_rate, sess, num_classes=14, is_training=True)

    # while True:



# with tf.device("/gpu:0"):
#     # Construct model
#     pred, pred_bhwd = custom_layers_unet.unet(x, training=True)
#     gt = tf.cast(tf.reshape(y,[-1]),tf.int32)
#     # Define loss and optimizer
#     loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
#     loss_map = tf.multiply(loss_map,tf.to_float(tf.not_equal(gt,0)))
#     cost = tf.reduce_mean(loss_map)
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#     # Initializing the variables
#     init = tf.global_variables_initializer()
#
#     #total steps
#     #total_steps = 100
#
# # Launch the graph
#     config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     epoch = 1
#
#     #config.gpu_options.allow_growth=True
#     with tf.Session(config=config) as sess:
#         sess.run(init)
#
#         print('This has been initialised')
#         step = 1
#
#         while 1:
#
#             img, label = SUNRGBD_dataset.get_random_shuffle(batch_size)
#             #print(img.shape)
#
#             #plt.show()
#             label = np.reshape(label,[-1])
#
#             decay = np.floor((epoch - 1) / 30)
#             learningRate = base_learning_rate *  np.power(0.95, decay)
#             #learning_rate = (1/step) * 0.1
#             _,lr = sess.run([optimizer,learning_rate], feed_dict={x: img, y: label, learning_rate:learningRate})
#             loss= sess.run([cost], feed_dict={x: img,y: label})
#             #print(lmap.shape)
#             print('epoch = ', epoch, 'batch = ', step-(np.floor(5285/batch_size))*(epoch-1), 'loss = ', loss, 'learning rate =', lr)
#             step = step + 1
#             epoch = np.floor(step*batch_size/5285)+1
#             tpred, tpred_bhwd = sess.run([pred, pred_bhwd], feed_dict={x: img,y: label})
#             #print(tpred_bhwd.shape)
#
#             best_labels = np.argmax(tpred_bhwd,axis=3)
#             #print(best_labels[1])
#             #print(best_labels.shape)
#
#             batchImage = tile_images(best_labels,batch_size, rows, cols, 1)
#             im.set_data(np.uint8(batchImage));
#             #print('max = ',img[1].max(),'min= ', img[1].min())
#             #im.set_clim(vmin=0.0, vmax=255.0)
#             fig.show();
#             pl.pause(0.00001);
  
    

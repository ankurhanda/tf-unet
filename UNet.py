import tensorflow as tf
import tensorflow.contrib.layers as layers
import resnet_utils
import numpy as np


class UNet(object):

    def __init__(self, batch_size, img_height, img_width, learning_rate, sess, num_classes=14, is_training=True):

        self.input_tensor = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])

        self.gt_labels = tf.placeholder(tf.int32, batch_size * img_width * img_height)
        self.gt = tf.cast(tf.reshape(self.gt_labels, [-1]), tf.int32)

        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.prediction, self.pred_classes = self.build_network(input_tensor= self.input_tensor,
                                             initializer=he_initializer,
                                             is_training=is_training,
                                             num_classes=14)

        self.flattened_pred = tf.reshape(self.prediction, [-1, num_classes])

        # Define loss and optimizer
        loss_map  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.flattened_pred, labels=self.gt)
        loss_map  = tf.multiply(loss_map, tf.to_float(tf.not_equal(self.gt, 0)))

        self.cost = tf.reduce_mean(loss_map)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.cost)

        self.sess = sess


    def build_network(self, input_tensor, initializer, is_training, num_classes=14):

        # with tf.variable_scope('encoder') as scope:

        conv_layer = layers.conv2d(input_tensor, num_outputs=64, kernel_size=(3,3),
                                   stride=1, padding='SAME', weights_initializer=initializer,
                                   activation_fn=tf.identity)
        conv_layer_enc_64 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3,3),
                                                       output_channels=64, initializer=initializer,
                                                       stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = layers.max_pool2d(inputs=conv_layer_enc_64, kernel_size=(2,2), stride=2)


        # Input size is H/2 x W/2
        conv_layer_enc_128 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                        output_channels=128, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_128 = resnet_utils.conv_bn_layer(conv_layer_enc_128, kernel_size=(3, 3),
                                                        output_channels=128, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = layers.max_pool2d(inputs=conv_layer_enc_128, kernel_size=(2, 2), stride=2)


        #Input size is H/4 x W/4
        conv_layer_enc_256 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                        output_channels=256, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_256 = resnet_utils.conv_bn_layer(conv_layer_enc_256, kernel_size=(3, 3),
                                                        output_channels=256, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = layers.max_pool2d(inputs=conv_layer_enc_256, kernel_size=(2, 2), stride=2)

        # Input size is H/8 x W/8
        conv_layer_enc_512 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                        output_channels=512, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_512 = resnet_utils.conv_bn_layer(conv_layer_enc_512, kernel_size=(3, 3),
                                                        output_channels=512, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = layers.max_pool2d(inputs=conv_layer_enc_512, kernel_size=(2, 2), stride=2)


        conv_layer_enc_1024 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                         output_channels=1024, initializer=initializer,
                                                         stride=1, bn=True, is_training=is_training, relu=True)

        # with tf.variable_scope('decoder') as scope:

        conv_layer_dec_512 = resnet_utils.conv_bn_layer(conv_layer_enc_1024, kernel_size=(3, 3),
                                                        output_channels=512, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)

        reduced_patchsize = np.multiply(conv_layer_dec_512.get_shape().as_list()[1:3], 2)
        conv_layer_dec_512 = tf.image.resize_images(conv_layer_dec_512, size=reduced_patchsize,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        conv_layer_dec_1024=  tf.concat([conv_layer_dec_512, conv_layer_enc_512], axis=3)
        conv_layer_dec_512 = resnet_utils.conv_bn_layer(conv_layer_dec_1024, kernel_size=(3, 3),
                                                        output_channels=512, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_dec_256 = resnet_utils.conv_bn_layer(conv_layer_dec_512, kernel_size=(3, 3),
                                                        output_channels=256, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)

        reduced_patchsize = np.multiply(conv_layer_dec_256.get_shape().as_list()[1:3], 2)
        conv_layer_dec_256 = tf.image.resize_images(conv_layer_dec_256, size=reduced_patchsize,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        conv_layer_dec_512 = tf.concat([conv_layer_dec_256, conv_layer_enc_256], axis=3)
        conv_layer_dec_256 = resnet_utils.conv_bn_layer(conv_layer_dec_512, kernel_size=(3, 3),
                                                        output_channels=256, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_dec_128 = resnet_utils.conv_bn_layer(conv_layer_dec_256, kernel_size=(3, 3),
                                                        output_channels=128, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)


        reduced_patchsize = np.multiply(conv_layer_dec_128.get_shape().as_list()[1:3], 2)
        conv_layer_dec_128 = tf.image.resize_images(conv_layer_dec_128, size=reduced_patchsize,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        conv_layer_dec_256 = tf.concat([conv_layer_dec_128, conv_layer_enc_128], axis=3)
        conv_layer_dec_128 = resnet_utils.conv_bn_layer(conv_layer_dec_256, kernel_size=(3, 3),
                                                        output_channels=128, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_dec_64 = resnet_utils.conv_bn_layer(conv_layer_dec_128, kernel_size=(3, 3),
                                                       output_channels=64, initializer=initializer,
                                                       stride=1, bn=True, is_training=is_training, relu=True)


        reduced_patchsize = np.multiply(conv_layer_dec_64.get_shape().as_list()[1:3], 2)
        conv_layer_dec_64 = tf.image.resize_images(conv_layer_dec_64, size=reduced_patchsize,
                                                   method=tf.image.ResizeMethod.BILINEAR)

        conv_layer_dec_128 = tf.concat([conv_layer_dec_64, conv_layer_enc_64], axis=3)
        conv_layer_dec_64 = resnet_utils.conv_bn_layer(conv_layer_dec_128, kernel_size=(3, 3),
                                                       output_channels=64, initializer=initializer,
                                                       stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_dec_64 = resnet_utils.conv_bn_layer(conv_layer_dec_64, kernel_size=(3, 3),
                                                       output_channels=64, initializer=initializer,
                                                       stride=1, bn=True, is_training=is_training, relu=True)

        prediction = layers.conv2d(conv_layer_dec_64, num_outputs=num_classes, kernel_size=(3,3),
                                   stride=1, padding='SAME', weights_initializer=initializer,
                                   activation_fn=tf.identity)

        classes = tf.cast(tf.argmax(prediction, 1), tf.uint8)

        return prediction, classes

    def train(self, inputs, labels):
        self.sess.run([self.train_op, self.cost], feed_dict={
            self.input_tensor: inputs,
            self.gt_labels: labels
        })

    def predict(self, inputs):
        return self.sess.run([self.prediction, self.pred_classes], feed_dict={
            self.input_tensor: inputs
        })
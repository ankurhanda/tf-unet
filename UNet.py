import tensorflow as tf
import tensorflow.contrib.layers as layers
import resnet_utils
import numpy as np

from collections import OrderedDict


class unet(object):

    def __init__(self, batch_size, img_height, img_width, learning_rate, sess, num_classes=14, is_training=True):

        self.sess = sess

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width  = img_width

        #for boot strapping loss
        self.K = self.img_width * 64

        self.input_tensor = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])

        self.gt_labels = tf.placeholder(tf.int32, batch_size * img_width * img_height)
        self.gt = tf.cast(tf.reshape(self.gt_labels, [-1]), tf.int32)
        he_initializer = tf.contrib.layers.variance_scaling_initializer()

        self.prediction, self.pred_classes, self.cost = self.build_network(initializer=he_initializer,
                                                                           is_training=is_training,
                                                                           num_classes=14)

        # self.prediction, self.pred_classes, self.cost = self.build_network_clean(initializer=he_initializer,
        #                                                                    is_training=is_training,
        #                                                                    num_classes=14)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
           self.train_op = optimizer.minimize(self.cost)


    def build_network_clean(self, initializer, is_training, num_classes=14):

        enc_layers = OrderedDict()
        dec_layers = OrderedDict()

        conv_layer = layers.conv2d(self.input_tensor, num_outputs=64, kernel_size=(3, 3),
                                   stride=1, padding='SAME', weights_initializer=initializer,
                                   activation_fn=tf.identity)

        enc_layers['conv_layer_enc_64'] = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                       output_channels=64, initializer=initializer,
                                                       stride=1, bn=True, is_training=is_training, relu=True)

        conv_layer = layers.max_pool2d(inputs=enc_layers['conv_layer_enc_64'], kernel_size=(2, 2), stride=2)

        for n_feat in [128, 256, 512]:

            enc_layers['conv_layer_enc_' + str(n_feat)] = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                            output_channels=n_feat, initializer=initializer,
                                                            stride=1, bn=True, is_training=is_training, relu=True)

            enc_layers['conv_layer_enc_' + str(n_feat)] = resnet_utils.conv_bn_layer(enc_layers['conv_layer_enc_' + str(n_feat)], kernel_size=(3, 3),
                                                            output_channels=n_feat, initializer=initializer,
                                                            stride=1, bn=True, is_training=is_training, relu=True)

            conv_layer = layers.max_pool2d(inputs=enc_layers['conv_layer_enc_' + str(n_feat)], kernel_size=(2, 2), stride=2)


        conv_layer_enc_1024 = resnet_utils.conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                                         output_channels=1024, initializer=initializer,
                                                         stride=1, bn=True, is_training=is_training, relu=True)
        dec_layers['conv_layer_dec_512'] = resnet_utils.conv_bn_layer(conv_layer_enc_1024, kernel_size=(3, 3),
                                                        output_channels=512, initializer=initializer,
                                                        stride=1, bn=True, is_training=is_training, relu=True)
        reduced_patchsize = np.multiply(dec_layers['conv_layer_dec_512'].get_shape().as_list()[1:3], 2)
        dec_layers['conv_layer_dec_512'] = tf.image.resize_images(dec_layers['conv_layer_dec_512'], size=reduced_patchsize,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        for n_feat in [512, 256, 128, 64]:

            dec_layers['conv_layer_dec_' + str(n_feat*2)] = tf.concat([dec_layers['conv_layer_dec_' + str(n_feat)],
                                                                       enc_layers['conv_layer_enc_' + str(n_feat)]],
                                                                                  axis=3)
            dec_layers['conv_layer_dec_' + str(n_feat)] = resnet_utils.conv_bn_layer(dec_layers['conv_layer_dec_' + str(n_feat*2)], kernel_size=(3, 3),
                                                            output_channels=n_feat, initializer=initializer,
                                                            stride=1, bn=True, is_training=is_training, relu=True)
            if n_feat > 64:
                dec_layers['conv_layer_dec_' + str(int(n_feat/2))] = resnet_utils.conv_bn_layer(dec_layers['conv_layer_dec_' + str(n_feat)], kernel_size=(3, 3),
                                                                output_channels=n_feat/2, initializer=initializer,
                                                                stride=1, bn=True, is_training=is_training, relu=True)

                reduced_patchsize = np.multiply(dec_layers['conv_layer_dec_' + str(int(n_feat/2))].get_shape().as_list()[1:3], 2)
                dec_layers['conv_layer_dec_' + str(int(n_feat/2))] = tf.image.resize_images(dec_layers['conv_layer_dec_' + str(int(n_feat/2))],
                                                                                         size=reduced_patchsize,
                                                                                         method=tf.image.ResizeMethod.BILINEAR)

        prediction = layers.conv2d(dec_layers['conv_layer_dec_64'], num_outputs=num_classes, kernel_size=(3, 3),
                                   stride=1, padding='SAME', weights_initializer=initializer,
                                   activation_fn=tf.identity)

        classes = tf.cast(tf.argmax(prediction, 1), tf.uint8)
        flattened_pred = tf.reshape(prediction, [-1, num_classes])

        # Define loss and optimizer
        loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flattened_pred, labels=self.gt)
        loss_map = tf.multiply(loss_map, tf.to_float(tf.not_equal(self.gt, 0)))

        # https://arxiv.org/pdf/1611.08323.pdf (Eq. 10)
        bootstrapping_loss, indices = tf.nn.top_k(tf.reshape(loss_map, [self.batch_size, self.img_height * self.img_width]),
                                   k=self.K, sorted=False)

        cost = tf.reduce_mean(tf.reduce_mean(bootstrapping_loss))

        return prediction, classes, cost

    def build_network(self, initializer, is_training, num_classes=14):

        conv_layer = layers.conv2d(self.input_tensor, num_outputs=64, kernel_size=(3,3),
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
        flattened_pred = tf.reshape(prediction, [-1, num_classes])

        # Define loss and optimizer
        loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flattened_pred, labels=self.gt)
        loss_map = tf.multiply(loss_map, tf.to_float(tf.not_equal(self.gt, 0)))

        # https://arxiv.org/pdf/1611.08323.pdf (Eq. 10)
        # bootstrapping_loss, indices = tf.nn.top_k(tf.reshape(loss_map, [self.batch_size, self.img_height * self.img_width]),
        #                            k=self.K, sorted=False)
        #
        # cost = tf.reduce_mean(tf.reduce_mean(bootstrapping_loss))

        cost = tf.reduce_mean(loss_map)

        return prediction, classes, cost

    def get_cost(self, inputs, labels):
        return self.sess.run(self.cost, feed_dict={
            self.input_tensor: inputs,
            self.gt_labels: labels
        })

    def train_batch(self, inputs, labels):
        return self.sess.run([self.train_op, self.cost, self.prediction], feed_dict={
            self.input_tensor: inputs,
            self.gt_labels: labels
        })

    def predict(self, inputs):
        return self.sess.run([self.prediction, self.pred_classes], feed_dict={
            self.input_tensor: inputs
        })

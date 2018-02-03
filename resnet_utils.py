import tensorflow as tf
import tensorflow.contrib.layers as layers



def conv_bn_layer(input_tensor, kernel_size,output_channels,
               initializer, stride=1, bn=False,
               is_training=True, relu=True):

    # with tf.variable_scope(name) as scope:
    conv_layer = layers.conv2d(inputs=input_tensor,
                               num_outputs=output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               activation_fn=tf.identity,
                               padding='SAME',
                               weights_initializer=initializer)
    if bn and relu:
        #How to use Batch Norm: https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/README_BATCHNORM.md

        #Why scale is false when using ReLU as the next activation
        #https://datascience.stackexchange.com/questions/22073/why-is-scale-parameter-on-batch-normalization-not-needed-on-relu/22127

        #Using fuse operation: https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        conv_layer = layers.batch_norm(inputs=conv_layer, center=True, scale=False, is_training=is_training, fused=True)
        conv_layer = tf.nn.relu(conv_layer)

    if bn and not relu:
        conv_layer = layers.batch_norm(inputs=conv_layer, center=True, scale=True, is_training=is_training)

    # print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
    return conv_layer

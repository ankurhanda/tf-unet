# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.





import numpy as np
import tensorflow as tf
from collections import OrderedDict

DATA_TYPE = tf.float32
VARIABLE_COUNTER = 0

def variable(name, shape, initializer,regularizer=None):
    global VARIABLE_COUNTER
    with tf.device('/cpu:0'):
        VARIABLE_COUNTER += np.prod(np.array(shape))
        return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=DATA_TYPE, trainable=True)

def conv_layer(input_tensor,name,kernel_size,output_channels,initializer,stride=1,bn=False,training=False,relu=True):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        conv_layer = tf.nn.bias_add(conv, biases)
        if bn:
            conv_layer = batch_norm_layer(conv_layer,scope,training)
        if relu:
            conv_layer = tf.nn.relu(conv_layer, name=scope.name)
    print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
    return conv_layer

def residual_block(input_tensor,name,kernel_size,output_channels,initializer,stride=1,bn=True,training=False):
    print('')
    print('Residual Block')
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        conv_output = conv_layer(input_tensor,'conv1',kernel_size,output_channels,initializer,stride=stride,bn=bn,training=training,relu=True)
        conv_output = conv_layer(conv_output,'conv2',kernel_size,output_channels,initializer,stride=1,bn=bn,training=training,relu=False)
        if stride != 1 or input_channels != output_channels:
            old_input_shape = input_tensor.get_shape().as_list()
            input_tensor = conv_layer(input_tensor,'projection',stride,output_channels,initializer,stride=stride,bn=False,training=training,relu=False)
            print('Projecting input {0} -> {1}'.format(old_input_shape,input_tensor.get_shape().as_list()))
        res_output = tf.nn.relu(input_tensor + conv_output,name=scope.name)
    print('')
    return res_output

def deconv_layer(input_tensor,name,kernel_size,output_channels,initializer,stride=1,bn=False,training=False,relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    output_shape = list(input_shape)
    output_shape[1] *= stride
    output_shape[2] *= stride
    output_shape[3] = output_channels
    with tf.variable_scope(name) as scope:
        kernel = variable('weights', [kernel_size, kernel_size, output_channels, input_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        deconv = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        deconv_layer = tf.nn.bias_add(deconv, biases)
        if bn:
            deconv_layer = batch_norm_layer(deconv_layer,scope,training)
        if relu:
            deconv_layer = tf.nn.relu(deconv_layer, name=scope.name)
    print('Deconv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),deconv_layer.get_shape().as_list()))
    return deconv_layer

def max_pooling(input_tensor,name,factor=2):
    pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
    print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),pool.get_shape().as_list()))
    return pool

def fully_connected_layer(input_tensor,name,output_channels,initializer,bn=False,training=False,relu=True):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        weights = variable('weights', [input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input_tensor,weights), biases, name=scope.name)
        if bn:
            fc = batch_norm_layer(fc,scope,training)
        if relu:
            fc = tf.nn.relu(bias, name=scope.name)
    print('Fully connected layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),fc.get_shape().as_list()))
    return fc

def batch_norm_layer(input_tensor,scope,training):
    return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)

def dropout_layer(input_tensor,keep_prob,training):
    if training:
        return tf.nn.dropout(input_tensor,keep_prob)
    return input_tensor

def concat_layer(input_tensor1,input_tensor2,axis=3):
    output = tf.concat(3,[input_tensor1,input_tensor2])
    input1_shape = input_tensor1.get_shape().as_list()
    input2_shape = input_tensor2.get_shape().as_list()
    output_shape = output.get_shape().as_list()
    print('Concat layer {0} and {1} -> {2}'.format(input1_shape,input2_shape,output_shape))
    return output

def flatten(input_tensor,name):
    batch_size = input_tensor.get_shape().as_list()[0]
    with tf.variable_scope(name) as scope:
        flat = tf.reshape(input_tensor, [batch_size,-1])
    print('Flatten layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),flat.get_shape().as_list()))
    return flat

def classification_inference(images,training=True):
    print('-'*30)
    print('Network Architecture')
    print('-'*30)
    global VARIABLE_COUNTER
    VARIABLE_COUNTER = 0
    layer_name_dict = {}
    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name

    NUM_CLASS = 3
    dropout_keep_prob = 0.5
    bn = True
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    x = images
    for i in range(2):
        x = conv_layer(x,layer_name('conv'),3,64,he_initializer,bn=bn,training=training)
    x = max_pooling(x,layer_name('pool'))
    x = flatten(x,layer_name('flatten'))
    x = fully_connected_layer(x,layer_name('fc'),4096,he_initializer,bn=bn,training=training)
    x = dropout_layer(x,dropout_keep_prob,training)
    x = fully_connected_layer(x,layer_name('fc'),NUM_CLASS,he_initializer,bn=False,training=training)
    print('-'*30)
    print('Number of variables:{0}'.format(VARIABLE_COUNTER))
    print('-'*30)
    print('')
    return x


def unet(images, training=True):
    print('-'*30)
    print('Network Architecture')
    print('-'*30)
    global VARIABLE_COUNTER
    VARIABLE_COUNTER = 0
    layer_name_dict = {}
    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name
        
        
    NUM_CLASS = 14
    bn = True
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    x = images  
    
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    #Build the network
    x = conv_layer(x,layer_name('conv'),3,64,he_initializer, bn = bn, training = training)
    dw_h_convs[0] = conv_layer(x,layer_name('conv'),3,64,he_initializer, bn = bn, training = training)
    x = max_pooling(dw_h_convs[0], 'pool1')
   
     
    dw_h_convs[1] = conv_layer(x,layer_name('conv'),3 ,128, he_initializer, bn = bn, training = training)
    dw_h_convs[1] = conv_layer(dw_h_convs[1],layer_name('conv'),3,128, he_initializer, bn = bn, training = training)
    dw_h_convs[2] = max_pooling(dw_h_convs[1],'pool2')
    
    
    dw_h_convs[2] = conv_layer(dw_h_convs[2],layer_name('conv'),3,256,he_initializer, bn = bn, training = training)
    dw_h_convs[2] = conv_layer(dw_h_convs[2],layer_name('conv'),3,256,he_initializer, bn = bn, training = training)
    dw_h_convs[3] = max_pooling(dw_h_convs[2],'pool3')
   
    
    dw_h_convs[3] = conv_layer(dw_h_convs[3],layer_name('conv'),3,512,he_initializer, bn = bn, training = training)
    dw_h_convs[3] = conv_layer(dw_h_convs[3],layer_name('conv'),3,512,he_initializer, bn = bn, training = training)
    dw_h_convs[4] = max_pooling(dw_h_convs[3],'pool4')
    
    
    
    dw_h_convs[4] = conv_layer(dw_h_convs[4],layer_name('conv'),3,1024, he_initializer, bn = bn, training = training) 
    dw_h_convs[4] = conv_layer(dw_h_convs[4],layer_name('conv'),3,512, he_initializer, bn = bn, training = training) 
        
    
    
    up_h_convs[0] = tf.image.resize_images(dw_h_convs[4], [ dw_h_convs[4].get_shape().as_list()[1]*2, 
                                                            dw_h_convs[4].get_shape().as_list()[2]*2] )  

    #print('size of up_h_convs[0] = ', up_h_convs[0].get_shape().as_list())
             
    up_h_convs[0] = tf.concat([up_h_convs[0], dw_h_convs[3] ],3 ) 
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 512, he_initializer, bn = bn, training = training)
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 256, he_initializer, bn = bn, training = training)
    
    up_h_convs[1] = tf.image.resize_images(up_h_convs[0], [ up_h_convs[0].get_shape().as_list()[1]*2, 
                                                            up_h_convs[0].get_shape().as_list()[2]*2] )  
    
    #print('size of up_h_convs[1] = ', up_h_convs[1].get_shape().as_list())    
    up_h_convs[1] = tf.concat([up_h_convs[1], dw_h_convs[2] ],3 ) 
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 256, he_initializer, bn = bn, training = training)
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 128, he_initializer, bn = bn, training = training)
    
    up_h_convs[2] = tf.image.resize_images(up_h_convs[1], [ up_h_convs[1].get_shape().as_list()[1]*2, 
                                                            up_h_convs[1].get_shape().as_list()[2]*2] )  

    #print('size of up_h_convs[0] = ', up_h_convs[2].get_shape().as_list())        
    up_h_convs[2] = tf.concat([up_h_convs[2], dw_h_convs[1] ],3 ) 
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 128, he_initializer, bn = bn, training = training)
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)

    up_h_convs[3] = tf.image.resize_images(up_h_convs[2], [ up_h_convs[2].get_shape().as_list()[1]*2, 
                                                            up_h_convs[2].get_shape().as_list()[2]*2] )
                                                            
    #print('size of up_h_convs[3] = ', up_h_convs[3].get_shape().as_list())                                                            
    #print('size of dw_h_convs[2] = ', dw_h_convs[2].get_shape().as_list())
    
    up_h_convs[3] = tf.concat([up_h_convs[3], dw_h_convs[0] ],3 ) 
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)
    

    #out = conv_layer(up_h_convs[0], layer_name('conv'), 1, 14, he_initializer, bn = False, training = training, relu=False)
    out = conv_layer(up_h_convs[3], layer_name('conv'), 1, 14, he_initializer, bn = False, training = training, relu=False)
    out_bhwd = out

        
    
    out = tf.reshape(out,[-1, NUM_CLASS])

    

    print('size of out= ', out.get_shape().as_list())

    return out, out_bhwd

def residual_inference(images,training=True):
    print('-'*30)
    print('Network Architecture')
    print('-'*30)
    global VARIABLE_COUNTER
    VARIABLE_COUNTER = 0
    layer_name_dict = {}
    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name

    NUM_CLASS = 14
    dropout_keep_prob = 0.5
    bn = True
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    x = images

    # Build and return the network
    for i in range(4):
        x = conv_layer(x,layer_name('conv'),3,64,he_initializer,bn=bn,training=training)
    x = residual_block(x,layer_name('resblock'),3,64,he_initializer,stride=2,bn=bn,training=training)
    for i in range(8):
        x = residual_block(x,layer_name('resblock'),3,64,he_initializer,bn=bn,training=training)
    x = residual_block(x,layer_name('resblock'),3,128,he_initializer,stride=2,bn=bn,training=training)
    for i in range(16):
        x = residual_block(x,layer_name('resblock'),3,128,he_initializer,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,128,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,64,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,NUM_CLASS,he_initializer,bn=False,training=training,relu=False)
    print('-'*30)
    print('Number of variables:{0}'.format(VARIABLE_COUNTER))
    print('-'*30)
    print('')
    return x

def inference(images,depths,training=True):
    print('-'*30)
    print('Network Architecture')
    print('-'*30)
    global VARIABLE_COUNTER
    VARIABLE_COUNTER = 0
    layer_name_dict = {}
    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name

    NUM_CLASS = 14
    dropout_keep_prob = 0.5
    bn = True
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    x = images
    y = depths

    # Build and return the network
    # RGB
    for i in range(3):
        x = conv_layer(x,layer_name('conv'),3,64,he_initializer,bn=bn,training=training)
    x = max_pooling(x,layer_name('max_pooling'))
    for i in range(3):
        x = conv_layer(x,layer_name('conv'),3,128,he_initializer,bn=bn,training=training)
    x = max_pooling(x,layer_name('max_pooling'))

    # Depth
    for i in range(3):
        y = conv_layer(y,layer_name('conv'),3,64,he_initializer,bn=bn,training=training)
    y = max_pooling(y,layer_name('max_pooling'))
    for i in range(3):
        y = conv_layer(y,layer_name('conv'),3,128,he_initializer,bn=bn,training=training)
    y = max_pooling(y,layer_name('max_pooling'))

    # Concat the two
    x = concat_layer(x,y)

    x = deconv_layer(x,layer_name('deconv'),3,128,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,64,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,NUM_CLASS,he_initializer,bn=False,training=training,relu=False)
    print('-'*30)
    print('Number of variables:{0}'.format(VARIABLE_COUNTER))
    print('-'*30)
    print('')
    return x

def loss(predictions, labels):
    num_classes = predictions.get_shape().as_list()[-1]
    flat_predictions = tf.reshape(predictions, [-1,num_classes])
    flat_labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(flat_predictions, flat_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    weight_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return tf.add(cross_entropy_mean,weight_loss)

def accuracy(predictions, labels):
    batch_size = predictions.get_shape().as_list()[0]
    arg_max_preds = tf.argmax(predictions, 3)
    flat_predictions = tf.reshape(arg_max_preds, [batch_size,-1])
    flat_labels = tf.reshape(labels, [batch_size,-1])
    correct_prediction = tf.equal(tf.cast(flat_predictions,tf.int32), flat_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

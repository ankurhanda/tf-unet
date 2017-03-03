# tf-unet
tensorflow version of unet

u-net is defined in the **custom_layers_unet.py** as a function 
```
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

           
    up_h_convs[0] = tf.concat([up_h_convs[0], dw_h_convs[3] ],3 ) 
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 512, he_initializer, bn = bn, training = training)
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 256, he_initializer, bn = bn, training = training)
    
    up_h_convs[1] = tf.image.resize_images(up_h_convs[0], [ up_h_convs[0].get_shape().as_list()[1]*2, 
                                                            up_h_convs[0].get_shape().as_list()[2]*2] )  
    
    up_h_convs[1] = tf.concat([up_h_convs[1], dw_h_convs[2] ],3 ) 
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 256, he_initializer, bn = bn, training = training)
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 128, he_initializer, bn = bn, training = training)
    
    up_h_convs[2] = tf.image.resize_images(up_h_convs[1], [ up_h_convs[1].get_shape().as_list()[1]*2, 
                                                            up_h_convs[1].get_shape().as_list()[2]*2] )  

    up_h_convs[2] = tf.concat([up_h_convs[2], dw_h_convs[1] ],3 ) 
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 128, he_initializer, bn = bn, training = training)
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)

    up_h_convs[3] = tf.image.resize_images(up_h_convs[2], [ up_h_convs[2].get_shape().as_list()[1]*2, 
                                                            up_h_convs[2].get_shape().as_list()[2]*2] )
      
    up_h_convs[3] = tf.concat([up_h_convs[3], dw_h_convs[0] ],3 ) 
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, he_initializer, bn = bn, training = training)
   
    out = conv_layer(up_h_convs[3], layer_name('conv'), 1, 14, he_initializer, bn = False, training = training, relu=False)
    out_bhwd = out

   
    out = tf.reshape(out,[-1, NUM_CLASS])

    print('size of out= ', out.get_shape().as_list())

    return out, out_bhwd
        
```
    
To make sure you don't cnosider the pixels with zero labels *i.e.* pxels with missing labels make sure to do the following
    
```
loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
loss_map = tf.multiply(loss_map,tf.to_float(tf.not_equal(gt,0)))
```

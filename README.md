# tf-unet
tensorflow version of unet

u-net is defined in the **UNet.py** 
    
To make sure you don't consider the pixels with zero labels *i.e.* pixels with missing labels, do the following
    
```Python
loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
loss_map = tf.multiply(loss_map,tf.to_float(tf.not_equal(gt,0)))
```
to train 
```Python
python3 train_u-net.py
``` 

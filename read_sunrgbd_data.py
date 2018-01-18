from __future__ import print_function

#import tensorflow as tf
#import custom_layers_unet
import os, sys
import numpy as np

from PIL import Image

from random import randint
#img = Image.open("/media/ankur/nnseg/sunrgbd_rgb/train/img-004123.jpg")
#img2 = img.resize((320,240),Image.NEAREST)

#img2.show()
#print(img.format, img.size, img.mode)

class dataset:
    def __init__(self, name, dataset_file):
        
        self.name = name
        self.dataset_file = dataset_file

        self.rgb_names = []
        self.label_names = []        
            
        i = 0    
        with open(dataset_file,"r") as f:
            for line in f:
                self.rgb_names.append(line.split()[0])
                self.label_names.append(line.split()[1])
                print(line.split()[0], line.split()[1])
                i+=1
                
        self.dataset_size = i 
        
    def get_random_shuffle(self, batch_size):
        
        imgarray   = np.empty([batch_size, 240, 320, 3],dtype=np.float32)
        labelarray = np.empty([batch_size, 240, 320],dtype=np.float32)
        
        for x in range(0,batch_size):
            rand_i = randint(1,self.dataset_size-5)
            img = Image.open(self.rgb_names[rand_i]).resize((320,240),Image.BILINEAR)
            labelImg = Image.open(self.label_names[rand_i]).resize((320,240),Image.NEAREST)
            imgarray[x] = np.asarray(img)
            labelarray[x] = np.asarray(labelImg)
            
        return imgarray,labelarray
    

#SUNRGBD_dataset = dataset("SUNRGBD","/media/ankur/nnseg/sunrgbd_training.txt")
#img, label = SUNRGBD_dataset.get_random_shuffle(4)
#Image.fromarray(np.uint8(img[1]),'RGB').show()
#label = np.reshape(label,[-1])
#print(label.shape)
    
        
    

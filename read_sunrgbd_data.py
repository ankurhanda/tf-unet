from __future__ import print_function

import os, sys, glob
import numpy as np
from PIL import Image
from random import randint

from tqdm import tqdm



class dataset:
    def __init__(self, name, dataset_file, img_type='rgb'):
        
        self.name = name
        self.dataset_file = dataset_file

        self.rgb_names = []
        self.label_names = []

        self.img_type = img_type

        if dataset_file.endswith('.txt'):
            i = 0
            with open(dataset_file,"r") as f:
                for line in tqdm(f):
                    self.rgb_names.append(line.split()[0])
                    self.label_names.append(line.split()[1])
                    # print(line.split()[0], line.split()[1])
                    i+=1
                    # if i == 100000:
                    #     break

            self.dataset_size = i
        else:
            depth_pngs = sorted(glob.glob(dataset_file +'/depth*.png'))
            label_pngs = sorted(glob.glob(dataset_file +'/label*.png'))

            assert len(depth_pngs) == len(label_pngs)

            for i in range(0, len(depth_pngs)):
                self.rgb_names.append(dataset_file + '/depthsub' + str(i) + '.png')
                self.label_names.append(dataset_file + '/labelssub' + str(i) + '.png')

            self.dataset_size = len(depth_pngs)

        self.shuffle_indices = list(range(0, self.dataset_size))
        np.random.shuffle(self.shuffle_indices)
        self.count = 0

        
    def get_random_shuffle(self, batch_size):

        if self.img_type == 'rgb':
            imgarray   = np.empty([batch_size, 240, 320, 3],dtype=np.float32)
        else:
            imgarray = np.empty([batch_size, 240, 320, 1], dtype=np.float32)

        labelarray = np.empty([batch_size, 240, 320],dtype=np.float32)
        
        for x in range(0,batch_size):

            rand_i = self.shuffle_indices[self.count]
            img = Image.open(self.rgb_names[rand_i]).resize((320,240),Image.BILINEAR)
            labelImg = Image.open(self.label_names[rand_i]).resize((320,240),Image.NEAREST)

            if self.img_type != 'rgb':
                imgarray[x] = np.expand_dims(np.asarray(img), axis=2)
            else:
                imgarray[x] = np.asarray(img)
            labelarray[x] = np.asarray(labelImg)

            self.count = self.count+1

            if self.count >= self.dataset_size:
                np.random.shuffle(self.shuffle_indices)
                self.count = 0

            
        return imgarray,labelarray
    

#SUNRGBD_dataset = dataset("SUNRGBD","/media/ankur/nnseg/sunrgbd_training.txt")
#img, label = SUNRGBD_dataset.get_random_shuffle(4)
#Image.fromarray(np.uint8(img[1]),'RGB').show()
#label = np.reshape(label,[-1])
#print(label.shape)
    
        
    

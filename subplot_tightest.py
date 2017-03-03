import matplotlib.pyplot as plt
import numpy as np
import argparse
import read_sunrgbd_data
import matplotlib as mpl


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
        


# Training settings
parser = argparse.ArgumentParser(description='plotting example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
                    
args = parser.parse_args()

rows = np.int(np.ceil(np.sqrt(args.batch_size)))
cols = np.int(np.ceil(args.batch_size / rows))

print('{0}, {1}'.format(rows,cols))

SUNRGBD_dataset = read_sunrgbd_data.dataset("SUNRGBD","/media/ankur/nnseg/sunrgbd_training.txt")
img, label = SUNRGBD_dataset.get_random_shuffle(args.batch_size)

#batchImages = tile_images(img, args.batch_size, rows, cols,3)
batchImages = tile_images(label, args.batch_size, rows, cols,1)

#inspired by http://jdherman.github.io/colormap/
colour_code = [(0, 0, 0),(0,0,1),(0.9137,0.3490,0.1882), (0, 0.8549, 0),
               (0.5843,0,0.9412),(0.8706,0.9451,0.0941),(1.0000,0.8078,0.8078),
               (0,0.8784,0.8980),(0.4157,0.5333,0.8000),(0.4588,0.1137,0.1608),
               (0.9412,0.1373,0.9216),(0,0.6549,0.6118),(0.9765,0.5451,0),
               (0.8824,0.8980,0.7608)];


cm = mpl.colors.ListedColormap(colour_code)

#{ {0 ,0 ,0},
#                {0, 0, 1}, --BED
#                {0.9137,0.3490,0.1882}, --BOOKS
#                {0, 0.8549, 0}, --CEILING
#                {0.5843,0,0.9412}, --CHAIR
#                {0.8706,0.9451,0.0941}, --FLOOR
#                {1.0000,0.8078,0.8078}, --FURNITURE
#                {0,0.8784,0.8980}, --OBJECTS
#                {0.4157,0.5333,0.8000}, --PAINTING
#                {0.4588,0.1137,0.1608}, --SOFA
#                {0.9412,0.1373,0.9216}, --TABLE
#                {0,0.6549,0.6118}, --TV
#                {0.9765,0.5451,0}, --WALL
#                {0.8824,0.8980,0.7608} --WINDOW
#              }

#Random Image
#someImage = np.random.random((240*np.int(rows),320*np.int(cols),3))


# Creare your figure and axes
fig, ax = plt.subplots(1)
someImage = np.random.random((240*np.int(rows),320*np.int(cols),14))
some_img_argmax = np.argmax(someImage,axis=2)

im1 = ax.imshow(np.uint8(some_img_argmax),extent=(0,1,1,0),cmap=cm)
ax.axis('tight')
ax.axis('off')

# Set whitespace to 0
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

for i in range(1,40):
    someImage = np.random.random((240*np.int(rows),320*np.int(cols),14))
    some_img_argmax = np.argmax(someImage,axis=2)
    im1.set_data(some_img_argmax)
    fig.show()
    plt.pause(0.00001);
    

# Display the image
#ax.imshow(np.uint8(batchImages),extent=(0,1,1,0))
# Display the labels

# Turn off axes and set axes limits
ax.axis('tight')
ax.axis('off')

plt.show()

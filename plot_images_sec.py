import matplotlib.pyplot as plt

img_per_sec = []
ngpus = 16

colours = ['red', 'orange', 'gold', 'deepskyblue', 'magenta', 'lightsalmon', 'blue']

p = 0

for ngpus in [8, 16, 32, 64, 96, 128]:
    img_per_sec = []
    
    for i in range(0,100):

        cur_num =0

        for gpu_id in range(0,ngpus):

            fileName = '../tf-summary-logs/time_gpus_{:03d}_gpuid_{:03d}_iter_{:03d}.txt'.format(ngpus, gpu_id, i)

            f = open(fileName, 'r')
            val = f.readline()
            f.close()

            cur_num += float(val)

        img_per_sec.append(cur_num)

    plt.semilogy(img_per_sec, c=colours[p], label=str(ngpus))
        
    p = p + 1

plt.legend(['8', '16', '32', '64', '96', '128'])
plt.ylabel('number of images processed per sec')
plt.xlabel('number of training iterations')
plt.show()

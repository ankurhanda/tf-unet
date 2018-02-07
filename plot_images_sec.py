import matplotlib.pyplot as plt

img_per_sec = []
ngpus = 16

for ngpus in [8, 16]:
    img_per_sec = []
    
    for i in range(0,100):

        cur_num =0

        for gpu_id in range(0,ngpus):

            fileName = '../tensorboard-results/time_gpus_{:03d}_gpuid_{:03d}_iter_{:03d}.txt'.format(ngpus, gpu_id, i)

            f = open(fileName, 'r')
            val = f.readline()
            f.close()

            cur_num += float(val)

        img_per_sec.append(cur_num)

        plt.plot(img_per_sec)

plt.ylabel('some numbers')
plt.show()

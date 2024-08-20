from torch.utils.tensorboard import SummaryWriter

import numpy as np

writer = SummaryWriter()
# writer = SummaryWriter(log_dir='./runs/version1')

# writer = SummaryWriter(comment='_resnet')

x = range(10)
r = 5

for i in range(10):
    # writer.add_scalar(tag='y=2x',
    #                   scalar_value=i * 2+10,
    #                   global_step=i)
    # writer.add_scalar('y=x^2', i ** 2+10, i)
    matrics = {'xsinx': i * np.sin(i / r),
               'xcosx': i * np.cos(i / r),
               'tanx': np.tan(i / r)}
    writer.add_scalars(main_tag="ch15",
                       tag_scalar_dict=matrics,
                       global_step=i)
    for k ,v in matrics.items():
        writer.add_scalar(tag=k,
                       scalar_value=v,
                       global_step=i)
    a = np.random.random(1000)
    writer.add_histogram(tag="distribution centers",
                         values=a*i,
                      global_step=i)

img = np.zeros((3,100,100))
img[0] = np.arange(0,10000).reshape(100,100)/10000
img[1] = 1- np.arange(0,10000).reshape(100,100)/10000
writer.add_image(tag='my_image',
                 img_tensor=img,
                 global_step=0,
                 dataformats='CHW')
img_HWC =  np.zeros((100,100,3))
img_HWC[:,:,0] = np.arange(0,10000).reshape(100,100)/10000
img_HWC[:,:,1] = 1 - np.arange(0,10000).reshape(100,100)/10000
writer.add_image(tag='my_image_HWC',
                 img_tensor=img_HWC,
                 global_step=0,
                 dataformats='HWC')

img_batch = np.zeros((16,3,100,100))
for i in range(16):
    img_batch[i,0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16*i
    img_batch[i,1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000 /16*i
writer.add_images(tag="my_image_batch",
                  img_tensor=img_batch,
                  global_step=0,
                  dataformats = "NCHW")
img1 = np.random.randn(8,100,100,1)
writer.add_images(tag="imgs",
                  img_tensor=img1,
                  global_step=0,
                  dataformats = "NHWC")
import cv2
name1 = "exp38"
file = f"/Users/zebulonzhang/deeplearning/Dog_breed/runs/train-cls/{name1}/validimages.jpg"
writer.add_image(tag=name1,
                 img_tensor=cv2.imread(file)[...,::-1],
                 global_step=0,
                 dataformats='HWC')

import matplotlib.image as mpimg
name2 = "resnet34"
file = f"/Users/zebulonzhang/deeplearning/Dog_breed/runs/train-cls/{name2}/validimages.jpg"
writer.add_image(tag=name2,
                 img_tensor=mpimg.imread(file),
                 global_step=0,
                 dataformats='HWC')
writer.close()


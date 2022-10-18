
from pdb import set_trace as pb
# from tf.image import adjust_hue

import tensorflow.compat.v2 as tf
import numpy as np
import torch
from  torchvision import transforms
from torchvision.transforms import functional_tensor as F_t
from albumentations.augmentations import functional as F_A

from check_crop_helper import *

np.random.seed(0)
imarray = np.random.rand(32,32,3)

imarray_tf = tf.convert_to_tensor(imarray)
imarray_pt = torch.tensor(imarray).T

pt_crop = transforms.RandomResizedCrop(
            32,
            scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
            ratio=(3.0/4.0, 4.0/3.0),
            # antialias=False
        )

tf_size = []
tf_ymax = []
tf_ymin = []
tf_xmax = []
tf_xmin = []
tf_ratio = []

pt_size = []
pt_ymax = []
pt_ymin = []
pt_xmax = []
pt_xmin = []
pt_ratio = []
# ================================================
for i in range(2000):
	bounding_boxes_tf = sample_box_tf(imarray_tf.numpy())
	tf_size.append(bounding_boxes_tf.shape[0]*bounding_boxes_tf.shape[1])
	tf_ymax.append(bounding_boxes_tf.shape[1])
	tf_xmax.append(bounding_boxes_tf.shape[0])
	tf_ratio.append(bounding_boxes_tf.shape[0]/bounding_boxes_tf.shape[1])
	bounding_boxes_pt = pt_crop.get_params(imarray_pt, pt_crop.scale, pt_crop.ratio)
	im = F_t.crop(imarray_pt, bounding_boxes_pt[0], bounding_boxes_pt[1], bounding_boxes_pt[2], bounding_boxes_pt[3])
	pt_size.append(im.shape[1]*im.shape[2])
	pt_ymax.append(im.shape[2])
	pt_xmax.append(im.shape[1])
	pt_ratio.append(im.shape[1]/im.shape[2])

print('size')
print(np.mean(tf_size))
print(np.mean(pt_size))
print('ymax')
print(np.mean(tf_ymax))
print(np.mean(pt_ymax))
print('xmax')
print(np.mean(tf_xmax))
print(np.mean(pt_xmax))
print('ratio')
print(np.mean(tf_ratio))
print(np.mean(pt_ratio))

print('size')
print(np.min(tf_size))
print(np.min(pt_size))
print('ymax')
print(np.min(tf_ymax))
print(np.min(pt_ymax))
print('xmax')
print(np.min(tf_xmax))
print(np.min(pt_xmax))
print('ratio')
print(np.min(tf_ratio))
print(np.min(pt_ratio))

print('size')
print(np.max(tf_size))
print(np.max(pt_size))
print('ymax')
print(np.max(tf_ymax))
print(np.max(pt_ymax))
print('xmax')
print(np.max(tf_xmax))
print(np.max(pt_xmax))
print('ratio')
print(np.max(tf_ratio))
print(np.max(pt_ratio))

pb()


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

pt_size = []
pt_ymax = []
pt_ymin = []
pt_xmax = []
pt_xmin = []
# ================================================
for i in range(2000):
	bounding_boxes_tf = sample_box_tf(imarray_tf)
	ymax = bounding_boxes_tf[1][1].numpy()
	ymin = bounding_boxes_tf[0][1].numpy()
	xmax = bounding_boxes_tf[1][0].numpy()
	xmin = bounding_boxes_tf[0][0].numpy()
	tf_size.append((xmax - xmin)*(ymax - ymin))
	tf_ymax.append(ymax)
	tf_ymin.append(ymin)
	tf_xmax.append(xmax)
	tf_xmin.append(xmin)
	bounding_boxes_pt = pt_crop.get_params(imarray_pt, pt_crop.scale, pt_crop.ratio)
	ymax = bounding_boxes_pt[3]
	ymin = bounding_boxes_pt[1]
	xmax = bounding_boxes_pt[2]
	xmin = bounding_boxes_pt[0]
	pt_size.append((xmax - xmin)*(ymax - ymin))
	pt_ymax.append(ymax)
	pt_ymin.append(ymin)
	pt_xmax.append(xmax)
	pt_xmin.append(xmin)

print('size')
print(np.mean(tf_size))
print(np.mean(pt_size))
print('ymax')
print(np.mean(tf_ymax))
print(np.mean(pt_ymax))
print('ymin')
print(np.mean(tf_ymin))
print(np.mean(pt_ymin))
print('xmax')
print(np.mean(tf_xmax))
print(np.mean(pt_xmax))
print('xmin')
print(np.mean(tf_xmin))
print(np.mean(pt_xmin))


pb()

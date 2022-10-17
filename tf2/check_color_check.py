
from pdb import set_trace as pb
# from tf.image import adjust_hue

import tensorflow.compat.v2 as tf
import numpy as np
import torch
from torchvision.transforms import functional_tensor as F_t
from albumentations.augmentations import functional as F_A

np.random.seed(0)
imarray = np.random.rand(100,100,3)

imarray_tf = tf.convert_to_tensor(imarray)
imarray_pt = torch.tensor(imarray)

factor = 1.1
# ==========================
# TF change brightness
# factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
#                            1.0 + max_delta)
image = tf.identity(imarray_tf) * factor
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change brightness
image_pt = F_t.adjust_brightness(imarray_pt.T.clone(), factor).T

#===================================
# albumentations change brightness
image_A = F_A.adjust_brightness_torchvision(imarray_pt.T.numpy().astype(np.float32), factor).T

print('brightness: pytorch vs tf')
print((np.abs(image_pt.numpy() - image.numpy()) > 1e-6).sum())
print('brightness: albumentations vs tf')
print((np.abs(image_A - image.numpy()) > 1e-6).sum())

# ==========================
# TF change hue
factor = 0.1
image = tf.image.adjust_hue(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change hue
image_pt = F_t.adjust_hue(imarray_pt.T.clone(), factor).T
#===================================
# albumentations change hue
image_A = F_A.adjust_hue_torchvision(imarray_pt.numpy().astype(np.float32), factor)

print('hue: pytorch vs tf')
print((np.abs(image_pt.numpy() - image.numpy()) > 1e-6).sum())
print('hue: albumentations vs tf')
print((np.abs(image_A - image.numpy()) > 1e-6).sum())
#===================================

# ==========================
# TF change contrast
factor = 0.9
image = tf.image.adjust_contrast(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change contrast
image_pt = F_t.adjust_contrast(imarray_pt.T.clone(), factor).T
#===================================
# albumentations change contrast
image_A = F_A.adjust_contrast_torchvision(imarray_pt.numpy().astype(np.float32), factor)

print('contrast: pytorch vs tf')
print((np.abs(image_pt.numpy() - image.numpy()) > 1e-6).sum())
print('contrast: albumentations vs tf')
print((np.abs(image_A - image.numpy()) > 1e-6).sum())
print('contrast: albumentations vs pytorch')
print((np.abs(image_A - image_pt.numpy()) > 1e-6).sum())
#===================================

# ==========================
# TF change saturation
factor = 1.1
image = tf.image.adjust_saturation(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change saturation
image_pt = F_t.adjust_saturation(imarray_pt.T.clone(), factor).T
#===================================
# albumentations change saturation
image_A = F_A.adjust_contrast_torchvision(imarray_pt.numpy().astype(np.float32), factor)

print('saturation: pytorch vs tf')
print((np.abs(image_pt.numpy() - image.numpy()) > 1e-6).sum())
print('saturation: albumentations vs tf')
print((np.abs(image_A - image.numpy()) > 1e-6).sum())
print('saturation: albumentations vs pytorch')
print((np.abs(image_A - image_pt.numpy()) > 1e-6).sum())
#===================================
pb()

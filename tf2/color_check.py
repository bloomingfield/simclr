
from pdb import set_trace as pb
# from tf.image import adjust_hue

import tensorflow.compat.v2 as tf
import numpy as np
import torch
from torchvision.transforms import functional_tensor as F_t

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

print(np.abs(image_pt.numpy() - image.numpy()).sum())

# ==========================
# TF change hue
factor = 0.1
image = tf.image.adjust_hue(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change hue
image_pt = F_t.adjust_hue(imarray_pt.T.clone(), factor).T

print(np.abs(image_pt.numpy() - image.numpy()).sum())
#===================================

# ==========================
# TF change contrast
factor = 0.9
image = tf.image.adjust_contrast(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change contrast
image_pt = F_t.adjust_contrast(imarray_pt.T.clone(), factor).T

print(np.abs(image_pt.numpy() - image.numpy()).sum())
#===================================

# ==========================
# TF change saturation
factor = 1.1
image = tf.image.adjust_saturation(tf.identity(imarray_tf), factor)
image = tf.clip_by_value(image, 0., 1.)
#===================================
# pytorch change saturation
image_pt = F_t.adjust_saturation(imarray_pt.T.clone(), factor).T

print(np.abs(image_pt.numpy() - image.numpy()).sum())
#===================================


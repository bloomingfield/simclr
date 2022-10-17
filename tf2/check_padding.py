
from pdb import set_trace as pb
# from tf.image import adjust_hue

import tensorflow.compat.v2 as tf
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class DiracInitializer(tf.keras.initializers.Initializer):

  def __init__(self, groups=1):
    self.groups = groups

  def __call__(self, shape, dtype=None, **kwargs):
    shape = tuple(reversed(shape))
    dimensions = len(shape)
    out_chans_per_grp = shape[0] // self.groups
    min_dim = min(out_chans_per_grp, shape[1])
    tensor = np.zeros(shape)
    for g in range(self.groups):
      for d in range(min_dim):
        if dimensions == 3:  # Temporal convolution
            tensor[g * out_chans_per_grp + d, d, tensor.shape[2] // 2] = 1
        elif dimensions == 4:  # Spatial convolution
            tensor[g * out_chans_per_grp + d, d, tensor.shape[2] // 2, tensor.shape[3] // 2] = 1
        else:  # Volumetric convolution
            tensor[g * out_chans_per_grp + d, d, tensor.shape(2) // 2,
                   tensor.shape[3] // 2, tensor.shape[4] // 2] = 1
    tensor = tensor.transpose()
    tensor = tf.convert_to_tensor(tensor, dtype=dtype)
    return tensor

  def get_config(self):  # To support serialization
    return {"groups": self.groups}

class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size, data_format='channels_last', **kwargs):
    super(FixedPadding, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs, training):
    kernel_size = self.kernel_size
    data_format = self.data_format
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               kernel_size,
               strides,
               data_format='channels_last',
               **kwargs):
    super(Conv2dFixedPadding, self).__init__(**kwargs)
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
    else:
      self.fixed_padding = None
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(),
        # kernel_initializer=tf.keras.initializers.Ones(),
        kernel_initializer=tf.keras.initializers.Constant(value=0.01),
        # kernel_initializer=DiracInitializer(),
        data_format=data_format)

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs, training=training)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, 
                     padding = (kernel_size - 1) // 2,
                     # padding = ('same' if stride == 1 else 'valid'), 
                     bias=bias)
    # padding = (kernel_size - 1) // 2
    # padding=('SAME' if stride == 1 else 'VALID')
    # variance_scaling_initializer(convolution.weight)
    nn.init.constant_(convolution.weight, 0.01)
    # nn.init.dirac_(convolution.weight)
    # torch.nn.init.kaiming_uniform_(convolution.weight, mode='fan_out', nonlinearity='relu')
    # nn.init.kaiming_normal_(convolution.weight, mode='fan_out', nonlinearity='relu')
    if convolution.bias is not None:
        torch.nn.init.zeros_(convolution.bias)
    return convolution

kernel =1 # either 1 3 or 7
stride = 1 # either 1 or 2

tf_conv = Conv2dFixedPadding(10,kernel,stride,data_format='channels_last')
py_conv = conv(3, 10, kernel, stride)


np.random.seed(0)
imarray = np.random.rand(1, 100,100, 3)

imarray_tf = tf.convert_to_tensor(imarray)
imarray_pt = torch.tensor(imarray)
imarray_pt = imarray_pt.squeeze().T[None, :, :, :].float()

res_tf = tf_conv(imarray_tf)
res_py = py_conv(imarray_pt).squeeze().T[None, :, :, :].float()

res = res_tf.numpy()-res_py.detach().numpy()


print((np.abs(res) > 1e-6).sum())

# pb()
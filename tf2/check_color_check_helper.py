
from pdb import set_trace as pb
# from tf.image import adjust_hue

from tensorflow.python.ops import gen_image_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
# https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/image_ops_impl.py#L2357

def convert_image_dtype(image, dtype, saturate=False, name=None):

  image = ops.convert_to_tensor(image, name='image')
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating and not dtype.is_integer:
    raise AttributeError('dtype must be either floating point or integer')
  if dtype == image.dtype:
    return array_ops.identity(image, name=name)

  with ops.name_scope(name, 'convert_image', [image]) as name:
    # Both integer: use integer multiplication in the larger range
    if image.dtype.is_integer and dtype.is_integer:
      scale_in = image.dtype.max
      scale_out = dtype.max
      if scale_in > scale_out:
        # Scaling down, scale first, then cast. The scaling factor will
        # cause in.max to be mapped to above out.max but below out.max+1,
        # so that the output is safely in the supported range.
        scale = (scale_in + 1) // (scale_out + 1)
        scaled = math_ops.floordiv(image, scale)

        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
      else:
        # Scaling up, cast first, then scale. The scale will not map in.max to
        # out.max, but converting back and forth should result in no change.
        if saturate:
          cast = math_ops.saturate_cast(image, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoring float overflows. If your image dynamic range
      # exceeds float range, you're on your own.
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        # Converting to float: first cast, then scale. No saturation possible.
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.multiply(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.multiply(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)

def adjust_contrast(images, contrast_factor):
  with ops.name_scope(None, 'adjust_contrast',
                      [images, contrast_factor]) as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype

    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_images = images
    else:
      flt_images = convert_image_dtype(images, dtypes.float32)

    adjusted = gen_image_ops.adjust_contrastv2(
        flt_images, contrast_factor=contrast_factor, name=name)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)

# def adjust_contrast2(images, contrast_factor):
#   images = math_ops.cast(images, dtypes.float32)
#   adjusted = gen_image_ops.adjust_contrastv2(
#       images, contrast_factor=contrast_factor)

#   return adjusted

def adjust_saturation(image, saturation_factor, name=None):
  with ops.name_scope(name, 'adjust_saturation', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    adjusted = gen_image_ops.adjust_saturation(flt_image, saturation_factor)

    return convert_image_dtype(adjusted, orig_dtype)

def sample_distorted_bounding_box_v2(image_size,
                                     bounding_boxes,
                                     seed=0,
                                     min_object_covered=0.1,
                                     aspect_ratio_range=None,
                                     area_range=None,
                                     max_attempts=None,
                                     use_image_if_no_bounding_boxes=None,
                                     name=None):
  if seed:
    seed1, seed2 = random_seed.get_seed(seed)
  else:
    if config.is_op_determinism_enabled():
      raise ValueError(
          f'tf.image.sample_distorted_bounding_box requires a non-zero seed to '
          f'be passed in when determinism is enabled, but got seed={seed}. '
          f'Please pass in a non-zero seed, e.g. by passing "seed=1".')
    seed1, seed2 = (0, 0)
  with ops.name_scope(name, 'sample_distorted_bounding_box'):
    return gen_image_ops.sample_distorted_bounding_box_v2(
        image_size,
        bounding_boxes,
        seed=seed1,
        seed2=seed2,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name) 


import torch
from PIL import Image
import numpy as np
import skimage.exposure
import os
import random
import torchvision.transforms as tfs


def total_variation(images, name=None):
    """
    Pytorch implementation of tf.image.total_variation for single channel images
    https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/ops/image_ops_impl.py#L3220-L3289

    :param images: Images of shape [batch, height, width]
    :param name: *Redundant*
    :return: Calculated total variation
    """
    ndims = len(list(images.shape))
    if ndims == 4:
        pixel_dif1 = images[:, 1:, :] - images[:, :-1, :]
        pixel_dif2 = images[:, :, 1:] - images[:, :, :-1]
        tot_var = (torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2)))
    else:
        raise ValueError('images must have 3 dimensions')
    return tot_var


def histogram_match(src):
    """
    Histogram match src to one of oasis dataset
    :param src: Image to match
    :return: Matched image
    """
    images = os.listdir('/home/Student/s4606685/summer_research/oasis-3/png_data/T1w-png-converted')
    ref = Image.open('/home/Student/s4606685/summer_research/oasis-3/png_data/T1w-png-converted/'+images[3000])
    ref = ref.convert('L')
    ref = ref.resize((128, 128))
    ref = np.array(ref)
    #ref = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
    matched = skimage.exposure.match_histograms(src, ref)
    return matched



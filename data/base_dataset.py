# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform(opt, grayscale=False, convert=True, crop=True, flip=True):
    """Create a torchvision transformation function

    The type of transformation is defined by option (e.g., [opt.preprocess], [opt.load_size], [opt.crop_size])
    and can be overwritten by arguments such as [convert], [crop], and [flip]

    Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        grayscale (bool)   -- if convert input RGB image to a grayscale image
        convert (bool)     -- if convert an image to a tensor array betwen [-1, 1]
        crop    (bool)     -- if apply cropping
        flip    (bool)     -- if apply horizontal flippling
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if opt.preprocess == 'resize_and_crop':
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.preprocess == 'crop' and crop:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.preprocess == 'scale_width':
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.crop_size)))
    elif opt.preprocess == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size)))
        if crop:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __adjust(img)))
    else:
        raise ValueError('--preprocess %s is not a valid option.' % opt.preprocess)

    if not opt.no_flip and flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __adjust(img):
    """Modify the width and height to be multiple of 4.

    Parameters:
        img (PIL image) -- input image

    Returns a modified image whose width and height are mulitple of 4.

    the size needs to be a multiple of 4,
    because going through generator network may change img size
    and eventually cause size mismatch error
    """
    ow, oh = img.size
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    """Resize images so that the width of the output image is the same as a target width

    Parameters:
        img (PIL image)    -- input image
        target_width (int) -- target image width

    Returns a modified image whose width matches the target image width;

    the size needs to be a multiple of 4,
    because going through generator network may change img size
    and eventually cause size mismatch error
    """
    ow, oh = img.size

    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

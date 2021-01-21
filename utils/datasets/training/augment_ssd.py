from random import shuffle as rand_shuffle, random as rand_random, uniform as rand_uniform, randint, \
    choice as rand_choice

from torch import FloatTensor, ones as torch_ones, float as torch_float, max as torch_max, min as torch_min
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, \
    hflip, resize, to_tensor, to_pil_image

from models.originals.ssd.external_modules import find_jaccard_overlap

"""Augmentation techniques as used in SSD rewrite in PyTorch.

This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
modifications are made. All credits to @sgrvinod.
"""


def photometric_distort(image):
    new_image = image

    distortions = [adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue]
    rand_shuffle(distortions)

    for distortion in distortions:
        if rand_random() < 0.5:
            if distortion.__name__ is 'adjust_hue':
                adjust_factor = rand_uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = rand_uniform(0.5, 1.5)

            new_image = distortion(new_image, adjust_factor)

    return new_image


def expand(image, boxes, filler):
    original_h = image.size(1)
    original_w = image.size(2)

    max_scale = 1.5
    scale = rand_uniform(1, max_scale)

    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    filler = FloatTensor(filler)
    new_image = torch_ones((3, new_h, new_w), dtype=torch_float) * filler.unsqueeze(1).unsqueeze(1)

    left = randint(0, new_w - original_w)
    right = left + original_w
    top = randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image
    new_boxes = boxes + FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    original_h = image.size(1)
    original_w = image.size(2)

    while True:
        min_overlap = rand_choice([0., .1, .3, .5, .7, .9, None])

        if min_overlap is None:
            return image, boxes, labels

        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = rand_uniform(min_scale, 1)
            scale_w = rand_uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left = randint(0, original_w - new_w)
            right = left + new_w
            top = randint(0, original_h - new_h)
            bottom = top + new_h
            crop = FloatTensor([left, top, right, bottom])

            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)

            if overlap.max().item() < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right]

            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.

            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)

            if not centers_in_crop.any():
                continue

            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            new_boxes[:, :2] = torch_max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch_min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    new_image = hflip(image)

    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize_and_maybe_transform_coordinates(image, boxes, dims=(300, 300), return_percent_coords=True):
    new_image = resize(image, dims)

    old_dims = FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims

    if not return_percent_coords:
        new_dims = FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, dims=(300, 300), do_flip=True, mean=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    new_image = photometric_distort(new_image)

    new_image = to_tensor(new_image)

    if rand_random() < 0.5:
        new_image, new_boxes = expand(new_image, boxes, filler=mean)
    new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

    new_image = to_pil_image(new_image)

    if do_flip and rand_random() < 0.5:
        new_image, new_boxes = flip(new_image, new_boxes)

    new_image, new_boxes = resize_and_maybe_transform_coordinates(new_image, new_boxes, dims=dims)

    return new_image, new_boxes, new_labels

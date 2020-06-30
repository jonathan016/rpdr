from random import uniform as rand_uniform, randint
import numpy as np

from PIL import Image

"""Augmentation techniques as used in YOLOv2 rewrite in PyTorch.

This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
names only. All credits to @marvis.
"""


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im


def rand_scale(s):
    scale = rand_uniform(1, s)
    if (randint(1, 10000) % 2):
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = rand_uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def augment(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = randint(-dw, dw)
    pright = randint(-dw, dw)
    ptop = randint(-dh, dh)
    pbot = randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    flip = randint(1, 10000) % 2
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


def fill_ground_truths(max_boxes, bs, flip, dx, dy, sx, sy):
    label = np.zeros((max_boxes, 5))

    bs = np.array(bs)
    bs = np.reshape(bs, (-1, 5))
    cc = 0
    for i in range(bs.shape[0]):
        x1 = bs[i][1] - bs[i][3] / 2
        y1 = bs[i][2] - bs[i][4] / 2
        x2 = bs[i][1] + bs[i][3] / 2
        y2 = bs[i][2] + bs[i][4] / 2

        x1 = min(0.999, max(0, x1 * sx - dx))
        y1 = min(0.999, max(0, y1 * sy - dy))
        x2 = min(0.999, max(0, x2 * sx - dx))
        y2 = min(0.999, max(0, y2 * sy - dy))

        bs[i][1] = (x1 + x2) / 2
        bs[i][2] = (y1 + y2) / 2
        bs[i][3] = (x2 - x1)
        bs[i][4] = (y2 - y1)

        if flip:
            bs[i][1] = 0.999 - bs[i][1]

        if bs[i][3] < 0.001 or bs[i][4] < 0.001:
            continue
        label[cc] = bs[i]
        cc += 1
        if cc >= max_boxes:
            break

    label = np.reshape(label, (-1))
    return label


def augment_data(image, max_object, yolo_labels, shape, jitter, hue, saturation, exposure):
    image, flip, dx, dy, sx, sy = augment(image, shape, jitter, hue, saturation, exposure)

    label = fill_ground_truths(max_object, yolo_labels, flip, dx, dy, 1. / sx, 1. / sy)

    return image, label

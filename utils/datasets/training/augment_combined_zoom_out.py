import os
from random import randint, random, choice as rand_choice, seed as rand_seed
from typing import Any, Dict
from warnings import warn

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ColorJitter, Grayscale, Normalize


def _generate_random_boxes_based_on_image_characteristics(total_boxes, images):
    min_width = min(map(lambda i: i.size[0], images))
    max_width = max(map(lambda i: i.size[0], images))
    min_height = min(map(lambda i: i.size[1], images))
    max_height = max(map(lambda i: i.size[1], images))

    random_boxes = []
    for _ in range(total_boxes):
        rand = rand_choice(['black_white', 'random_color', 'salt_pepper'])

        random_box_width = randint(min_width, max_width)
        random_box_height = randint(min_height, max_height)
        random_box_size = (max(random_box_width + randint(-50, 50), random_box_width),
                           max(random_box_height + randint(-50, 50), random_box_height))
        if rand == 'black_white':
            random_box = Image.new('RGB', random_box_size, rand_choice([(1, 1, 1), (255, 255, 255)]))
        elif rand == 'random_color':
            random_box = Image.new('RGB', random_box_size, (randint(0, 255), randint(0, 255), randint(0, 255)))
        else:
            random_box = Image.fromarray(np.random.randint(0, 256, (*random_box_size, 3), dtype=np.uint8))
        random_boxes.append(random_box)

    return random_boxes


def _varying_background_noise(combined_image_size, images, product_noise_sources=None, background_sources=None):
    if background_sources:
        background = background_sources[randint(0, len(background_sources) - 1)][0]
        background = background.resize(combined_image_size)
        background = background.convert('RGB')
    else:
        rand = rand_choice(['black_white', 'mean', 'random_color', 'salt_pepper'])

        if rand == 'black_white':
            background = Image.new('RGB', combined_image_size, rand_choice([(1, 1, 1), (255, 255, 255)]))
        elif rand == 'mean':
            total_pixels = sum(map(lambda i: i.size[0] * i.size[1], images))

            red = int(round(sum(map(lambda i: np.array(i)[:, :, 0].sum(), images)) / total_pixels))
            green = int(round(sum(map(lambda i: np.array(i)[:, :, 1].sum(), images)) / total_pixels))
            blue = int(round(sum(map(lambda i: np.array(i)[:, :, 2].sum(), images)) / total_pixels))

            background = Image.new('RGB', combined_image_size, (red, green, blue))
        elif rand == 'random_color':
            background = Image.new('RGB', combined_image_size, (randint(0, 255), randint(0, 255), randint(0, 255)))
        else:
            background = Image.fromarray(np.random.randint(0, 256, (*combined_image_size, 3), dtype=np.uint8))

    if randint(0, 2):
        random_boxes = _generate_random_boxes_based_on_image_characteristics(randint(2, 15), images)
        for box in random_boxes:
            left = randint(0, combined_image_size[0])
            upper = randint(0, combined_image_size[1])
            right = left + box.size[0]
            lower = upper + box.size[1]
            background.paste(box, (left, upper, right, lower))

    if product_noise_sources:
        def resize_maintain_aspect_ratio(im):
            width, height = image.size
            aspect_ratio = width / height
            has_looped_for = 0
            while has_looped_for < 1000:
                new_width, new_height = randint(20, im.width), randint(20, im.height)
                if abs((new_width / new_height) - aspect_ratio) < .2:
                    break
                has_looped_for += 1
            if has_looped_for == 1000:
                random_scale = random() + randint(0, 3)
                random_scale = random_scale if random_scale > .25 else .25
                new_width, new_height = int(width * random_scale), int(height * random_scale)

            return Resize((new_height, new_width))(image)

        background = background.convert('RGBA')
        total_images = randint(0, randint(15, 40))
        for i in range(total_images):
            image = product_noise_sources[randint(0, len(product_noise_sources) - 1)][0]

            image = resize_maintain_aspect_ratio(image)

            left = randint(0, combined_image_size[0])
            upper = randint(0, combined_image_size[1])
            right = left + image.size[0]
            lower = upper + image.size[1]
            if image.mode == 'RGBA':
                background.paste(image, (left, upper, right, lower), mask=image)
            else:
                background.paste(image, (left, upper, right, lower))

    return background


def _combine(identifiers, images, transform=None, min_resize=None, max_resize=None, random_seed=None, give_noise=False):
    if random_seed:
        rand_seed(random_seed)

    if transform:
        if min_resize and max_resize:
            def resize_maintain_aspect_ratio(image):
                width, height = image.size
                aspect_ratio = width / height
                has_looped_for = 0
                while has_looped_for < 1000:
                    new_width, new_height = randint(min_resize, max_resize), randint(min_resize, max_resize)
                    if abs((new_width / new_height) - aspect_ratio) < .2:
                        break
                    has_looped_for += 1
                if has_looped_for == 1000:
                    random_scale = random() + randint(0, 3)
                    random_scale = random_scale if random_scale > .25 else .25
                    new_width, new_height = int(width * random_scale), int(height * random_scale)

                return Resize((new_height, new_width))(image)

            images = map(resize_maintain_aspect_ratio, images)
        images = [transform(image) for image in images]

    identifiers_images = list(zip(identifiers, images))
    backup_identifiers_images = list(zip(identifiers, images))

    combined_image_size = _get_combined_image_size(images)
    if give_noise == 'varying':
        combined_image = _varying_background_noise(combined_image_size, images)
    elif type(give_noise) == tuple:
        background_noise, background_sources = give_noise
        combined_image = _varying_background_noise(combined_image_size, images, ImageFolder(root=str(
            background_noise), loader=Image.open), ImageFolder(root=str(background_sources), loader=Image.open))
    elif os.path.exists(give_noise) and os.path.isdir(give_noise):
        combined_image = _varying_background_noise(combined_image_size, images, ImageFolder(root=str(
            give_noise), loader=Image.open))
    elif give_noise:
        combined_image = Image.fromarray(np.random.randint(0, 256, (*combined_image_size, 3), dtype=np.uint8))
    else:
        combined_image = Image.new('RGBA', combined_image_size)
    coordinates = {k: None for k, _ in identifiers_images}

    total_loop = 0
    while len(identifiers_images) != 0:
        # Total loop approaching 1000 means current combined image cannot fit all the images; a 'restart' is required.
        # This is to prevent infinite loop.
        if total_loop == 1000:
            identifiers_images = backup_identifiers_images
            combined_image_size = _get_combined_image_size(images)
            if give_noise:
                combined_image = Image.fromarray(np.random.randint(0, 256, (*combined_image_size, 3), dtype=np.uint8))
            else:
                combined_image = Image.new('RGBA', combined_image_size)
            coordinates = {k: None for k, _ in identifiers_images}
            total_loop = 0

        for i in identifiers_images:
            identifier, image = i
            image_coordinate = _get_image_coordinate(image, combined_image_size)

            if _not_overlapping(coordinates, image_coordinate):
                combined_image.paste(image.convert('RGBA'), image_coordinate, image.convert('RGBA'))
                coordinates[identifier] = image_coordinate
                identifiers_images.remove(i)
        total_loop += 1

    return combined_image.convert('RGB'), coordinates


def _get_image_coordinate(image, base_image_size):
    image_width, image_height = image.size

    left, right = _get_safe_boundary(image_width, base_image_size[0])
    upper, lower = _get_safe_boundary(image_height, base_image_size[1])

    return left, upper, right, lower


def _get_safe_boundary(image_aspect, base_image_aspect):
    first = randint(0, base_image_aspect)
    second = first + image_aspect

    while second > base_image_aspect:
        first = randint(0, base_image_aspect)
        second = first + image_aspect

    return first, second


def _not_overlapping(coordinates, image_coordinate) -> bool:
    no_overlap = [_get_intersection(coordinate, image_coordinate) == 0.0 for coordinate in coordinates.values()]

    return all(no_overlap)


def _get_intersection(box1, box2):
    if box1 is None:
        return 0

    x = min(box1[0], box2[0]), max(box1[2], box2[2])
    y = min(box1[1], box2[1]), max(box1[3], box2[3])

    box1 = box1[2] - box1[0], box1[3] - box1[1]
    box2 = box2[2] - box2[0], box2[3] - box2[1]

    raw_union_width = x[1] - x[0]
    raw_union_height = y[1] - y[0]
    intersection_width = box1[0] + box2[0] - raw_union_width
    intersection_height = box1[1] + box2[1] - raw_union_height

    intersection = intersection_width * intersection_height if intersection_width > 0 and intersection_height > 0 else 0
    return float(intersection)


def _get_combined_image_size(images):
    sum_width = sum([x.size[0] for x in images])
    sum_height = sum([x.size[1] for x in images])

    max_resolution = max(sum_width + randint(0, sum_width // 4), sum_height + randint(0, sum_height // 4))
    combined_image_size = (max_resolution, max_resolution)

    return combined_image_size


def _combine_from_path(source_labels, transform=None, min_resize=None, max_resize=None, random_seed=None,
                       give_noise=False):
    images = [Image.open(path) for path in source_labels.values()]
    identifiers = [identifier for identifier in source_labels.keys()]

    return _combine(identifiers, images, transform, min_resize, max_resize, random_seed, give_noise)


def combined_zoom_out(source: Dict[Any, str] = None, identifiers: list = None, images: list = None,
                      individual_transform=None, min_resize: int = None, max_resize: int = None, seed: int = None,
                      give_noise: bool = False):
    """The proposed augmentation technique in this project's proposal. ``source`` is mutually exclusive to
    ``identifiers`` and ``images``, and vice versa.

    The idea is to accept a list of images, where each image contains only an object, and then combined to a single
    image to simulate the typical object detection input. A transformation can be applied for each image,
    with addition of randomized resizing. This method also generates and returns the coordinates for each image in
    the combined image to be used for further processing.

    :param source: A dictionary where the key is an identifier of any hashable type and value of image path (string).
    :param identifiers: Identifiers of ``images`` to get bounding boxes of the corresponding image after augmentation.
    :param images: List of ``PIL Image``s, each is ordered according to the order of ``identifiers``.
    :param individual_transform: Transformations to be applied to each image in ``images``. Must be from
      ``torchvision.transforms`` module and ideally should not contain any color changing transformation. For color
      changing transformation, set them on the ``combined_transform`` parameter.
    :param min_resize: Minimum resized image's width and/or height.
    :param max_resize: Maximum resized image's width and/or height.
    :param seed: Seed for the randomized coordinate and image resizing value randomization.
    :param give_noise: Flag whether to give background noise or not in the combined image.
    :return: A tuple of combined image in RGB format and a coordinate dictionary, where the key is the specified
      identifiers and the value is a tuple of (left, upper, right, lower) coordinate of each image's bounding box in
      the combined image.
    """

    if source and (identifiers or images):
        raise ValueError('source must be defined without identifiers and images')
    if (identifiers and images) and source:
        raise ValueError('identifiers and images must be defined without source')

    __check_for_color_changing_transform(individual_transform)

    combined_image, bounding_boxes = None, None
    if source:
        combined_image, bounding_boxes = _combine_from_path(
            source, individual_transform, min_resize, max_resize, seed, give_noise)
    elif identifiers and images:
        assert len(identifiers) == len(images)
        combined_image, bounding_boxes = _combine(
            identifiers, images, individual_transform, min_resize, max_resize, seed, give_noise)

    return combined_image, bounding_boxes


def __check_for_color_changing_transform(individual_transform):
    color_changing_transformations = [ColorJitter, Grayscale, Normalize]

    if type(individual_transform) is Compose:
        if any([type(t) in color_changing_transformations for t in individual_transform.transforms]):
            warn('Individual image transform should not contain color changing transformation(s)')
    elif type(individual_transform) in color_changing_transformations:
        warn('Individual image transform should not contain color changing transformation(s)')

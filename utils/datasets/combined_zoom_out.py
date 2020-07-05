from typing import Any, Dict
from warnings import warn

from PIL import Image
from random import randint, seed as rand_seed
from torchvision.transforms import Resize, Compose, ColorJitter, Grayscale, Normalize


def _combine(identifiers, images, transform=None, min_resize=None, max_resize=None, random_seed=None):
    if random_seed:
        rand_seed(random_seed)

    if transform:
        if min_resize and max_resize:
            images = [Resize((randint(min_resize, max_resize), randint(min_resize, max_resize)))(image) for image in
                      images]
        images = [transform(image) for image in images]

    identifiers_images = list(zip(identifiers, images))
    backup_identifiers_images = list(zip(identifiers, images))

    combined_image_size = _get_combined_image_size(images)
    combined_image = Image.new('RGBA', combined_image_size)
    coordinates = {k: None for k, _ in identifiers_images}

    total_loop = 0
    while len(identifiers_images) != 0:
        # Total loop approaching 1000 means current combined image cannot fit all the images; a 'restart' is required.
        # This is to prevent infinite loop.
        if total_loop == 1000:
            identifiers_images = backup_identifiers_images
            combined_image_size = _get_combined_image_size(images)
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

    combined_image_size = (sum_width + randint(0, sum_width), sum_height + randint(0, sum_height))

    return combined_image_size


def _combine_from_path(source_labels, transform=None, min_resize=None, max_resize=None, random_seed=None):
    images = [Image.open(path) for path in source_labels.values()]
    identifiers = [identifier for identifier in source_labels.keys()]

    return _combine(identifiers, images, transform, min_resize, max_resize, random_seed)


def combined_zoom_out(source: Dict[Any, str] = None, identifiers: list = None, images: list = None,
                      individual_transform=None, min_resize: int = None, max_resize: int = None, seed: int = None):
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
        combined_image, bounding_boxes = _combine_from_path(source, individual_transform, min_resize, max_resize, seed)
    elif identifiers and images:
        assert len(identifiers) == len(images)
        combined_image, bounding_boxes = _combine(identifiers, images, individual_transform, min_resize, max_resize,
                                                  seed)

    return combined_image, bounding_boxes


def __check_for_color_changing_transform(individual_transform):
    color_changing_transformations = [ColorJitter, Grayscale, Normalize]

    if type(individual_transform) is Compose:
        if any([type(t) in color_changing_transformations for t in individual_transform.transforms]):
            warn('Individual image transform should not contain color changing transformation(s)')
    elif type(individual_transform) in color_changing_transformations:
        warn('Individual image transform should not contain color changing transformation(s)')

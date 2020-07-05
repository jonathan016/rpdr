from random import sample as rand_sample, randint
from torch import FloatTensor, LongTensor, stack as torch_stack
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import ToTensor

from utils.datasets import combined_zoom_out
from utils.datasets.yolo_augmentation import augment_data as yolo_transform
from utils.datasets.ssd_augmentation import transform as ssd_transform


class GroZiDetectionDataset(ImageFolder):
    """Encapsulates GroZi-120 dataset with combined zoom out augmentation as proposed in this project's proposal.

    This dataset wrapper is usable for SSD and YOLO (v2 and v3) models only for now. After selecting
    ``max_data_in_combined`` individual product images, the selected images are combined with combined zoom out
    augmentation and later formatted to this dataset's specified model's format.

    Arguments:
        root (string): Root folder of the individual product images.
        transform (callable, optional): Transformations from ``torchvision.transforms`` module to be applied to each
            individual product images prior to combining.
        target_transform (callable, optional): Not directly used, but will be used by ``ImageFolder``'s
            implementation to transform the target values of the dataset.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an ``Image`` file
            and check if the file is a valid_file (used to check of corrupt files).
        transform_to_tensor (bool, optional): A flag to denote whether the combined image should be converted to
            tensor or not.
        min_resize (int, optional): The minimum resize width/height of each individual product image.
        max_resize (int, optional): The maximum resize width/height of each individual product image.
        max_data_usage (int, optional): Maximum individual product image usage in the combined zoom out augmentation
            technique.
        max_data_in_combined (int, optional): Maximum present individual product image in the combined image.
        max_object (int, optional): Number of possible objects in the dataset. Only to be specified if ``model`` is
            ``GroZiDetectionDataset.YOLO``.
        seen_images (int, optional): Number of seen images of the YOLO model. Useful for changing image size (
            multi-scale training) for YOLO models. Only to be specified if ``model`` is ``GroZiDetectionDataset.YOLO``.
        batch_size (int, optional): Batch size of training data. Useful for changing image size (
            multi-scale training) for YOLO models. Only to be specified if ``model`` is ``GroZiDetectionDataset.YOLO``.
    """
    SSD: str = 'SSD'
    YOLO: str = 'YOLO'

    def __init__(self, root, model: str = SSD, transform=None, target_transform=None, loader=default_loader,
                 is_valid_file=None, transform_to_tensor: bool = True, min_resize: int = None, max_resize: int = None,
                 max_data_usage: int = 1, max_data_in_combined: int = 1, max_object: int = None, seen_images: int =
                 0, batch_size: int = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        assert model == self.YOLO or model == self.SSD
        self.model = model
        if self.model == self.YOLO:
            assert max_object is not None and max_object > 0
            assert seen_images is not None and seen_images >= 0
            assert batch_size is not None and batch_size > 0
            self.max_object = max_object
            self.seen_images = seen_images
            self.batch_size = batch_size

        self.max_data_usage = max_data_usage
        self.max_data_in_combined = max_data_in_combined
        self.kwargs = {
            'min_resize': min_resize,
            'max_resize': max_resize,
            'individual_transform': self.transform
        }

        self.transform_to_tensor = transform_to_tensor
        self.usage = {k: 0 for k in self.samples}

    def __len__(self):
        return len(self.samples) * self.max_data_usage // self.max_data_in_combined

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError

        used_samples = self.__get_used_samples()

        images = []
        index_labels = []
        for index, sample in enumerate(used_samples):
            path, label = sample
            images.append(self.loader(path))
            # index is added since bounding_boxes keys will have overlapping values due to same label being in the
            # same image if index is not added. By grouping index and label to a tuple, a unique key is guaranteed
            # and no overlapping value in bounding_boxes dictionary is possible even with products of the same label
            # occurring more than once in the combined_image.
            index_labels.append((index, label))
            self.usage[(path, label)] += 1

        combined_image, bounding_boxes = combined_zoom_out(identifiers=index_labels, images=images, **self.kwargs)

        labels = [label for _, label in bounding_boxes.keys()]
        bounding_boxes = list(bounding_boxes.values())

        target = None
        if self.model == self.SSD:
            combined_image, *target = self._format_ssd(combined_image, labels, bounding_boxes)
        elif self.model == self.YOLO:
            combined_image, target = self._format_yolo(combined_image, labels, bounding_boxes)

        if self.transform_to_tensor:
            combined_image = ToTensor()(combined_image)

        if self.model == self.SSD:
            return (combined_image, *target)
        elif self.model == self.YOLO:
            return combined_image, target

    def _format_yolo(self, combined_image, labels, bounding_boxes):
        assert len(labels) == len(bounding_boxes)

        yolo_labels = []
        for i in range(len(labels)):
            cx, cy, w, h = GroZiDetectionDataset._get_center_size_coordinates(bounding_boxes[i], combined_image.size)
            yolo_labels.append([int(labels[i]), cx, cy, w, h])

        shape = self._get_shape_for_multi_scale_training()

        return yolo_transform(combined_image, self.max_object, yolo_labels, shape, .2, .1, 1.5, 1.5)

    def _get_shape_for_multi_scale_training(self):
        if self.seen_images < (4000 * self.batch_size):
            resolution = 416
        elif self.seen_images < (8000 * self.batch_size):
            resolution = (randint(0, 3) + 13) * 32
        elif self.seen_images < (12000 * self.batch_size):
            resolution = (randint(0, 5) + 12) * 32
        elif self.seen_images < (16000 * self.batch_size):
            resolution = (randint(0, 7) + 11) * 32
        else:
            resolution = (randint(0, 9) + 10) * 32

        return resolution, resolution

    @staticmethod
    def _get_center_size_coordinates(bounding_boxes, image_size):
        x_left, y_upper, x_right, y_lower = bounding_boxes
        image_width, image_height = image_size

        cx = (((x_right - x_left) / 2) + x_left) / image_width
        cy = (((y_lower - y_upper) / 2) + y_upper) / image_height
        w = (x_right - x_left) / image_width
        h = (y_lower - y_upper) / image_height

        return cx, cy, w, h

    @staticmethod
    def _format_ssd(combined_image, labels, bounding_boxes):
        labels = LongTensor(labels)
        bounding_boxes = FloatTensor(bounding_boxes)

        return ssd_transform(combined_image, bounding_boxes, labels)

    def __get_used_samples(self):
        indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
        used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
        fully_used = [self.usage[k] == self.max_data_usage for k in used_samples]

        while all(fully_used):
            indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
            used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
            fully_used = [self.usage[k] == 20 for k in used_samples]

        return used_samples


def ssd_collate(batch):
    """``DataLoader`` collation function for ``batch_size`` larger than 1.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
    modifications are made. All credits to @sgrvinod.
    """

    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch_stack(images, dim=0)

    return images, boxes, labels

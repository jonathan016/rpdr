from random import sample as rand_sample, randint
from torch import FloatTensor, LongTensor, stack as torch_stack
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import ToTensor

from utils.datasets import combined_zoom_out
from utils.datasets.yolo_augmentation import augment_data


class GroZiDetectionDataset(ImageFolder):
    SSD: str = 'SSD'
    YOLO: str = 'YOLO'

    def __init__(self, root, model: str = SSD, transform=None, target_transform=None, loader=default_loader,
                 is_valid_file=None, transform_to_tensor: bool = True, min_resize: int = None, max_resize: int = None,
                 combined_transform=None, max_data_usage: int = 1, max_data_in_combined: int = 1,
                 max_object: int = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        assert model == self.YOLO or model == self.SSD
        self.model = model
        if self.model == self.YOLO:
            assert max_object is not None
            self.max_object = max_object

        self.max_data_usage = max_data_usage
        self.max_data_in_combined = max_data_in_combined
        self.kwargs = {
            'min_resize': min_resize,
            'max_resize': max_resize,
            'individual_transform': self.transform,
            'combined_transform': combined_transform
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

        target = None
        if self.model == self.SSD:
            target = self._format_ssd(bounding_boxes)
        elif self.model == self.YOLO:
            combined_image, target = self._format_yolo(combined_image, bounding_boxes)

        if self.transform_to_tensor:
            combined_image = ToTensor()(combined_image)

        if self.model == self.SSD:
            return (combined_image, *target)
        elif self.model == self.YOLO:
            return combined_image, target

    def _format_yolo(self, combined_image, bounding_boxes):
        labels = [label for _, label in bounding_boxes.keys()]
        bounding_boxes = list(bounding_boxes.values())

        assert len(labels) == len(bounding_boxes)
        yolo_labels = []
        for i in range(len(labels)):
            cx, cy, w, h = GroZiDetectionDataset._get_center_size_coordinates(bounding_boxes[i], combined_image.size)
            yolo_labels.append([int(labels[i]), cx, cy, w, h])

        return augment_data(combined_image, self.max_object, yolo_labels, combined_image.size, .2, .1, 1.5, 1.5)

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
    def _format_ssd(bounding_boxes):
        labels = LongTensor([label for _, label in bounding_boxes.keys()])
        bounding_boxes = FloatTensor(list(bounding_boxes.values()))

        return bounding_boxes, labels

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
    """DataLoader collation function for batch_size larger than 1.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
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

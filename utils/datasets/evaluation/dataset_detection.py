import json
import os
from random import Random, randint, sample as rand_sample

from pandas import DataFrame
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor

from utils.datasets.training.augment_combined_zoom_out import combined_zoom_out
from utils.datasets.training.augment_ssd import transform as ssd_transform


class DetectionRecognitionDataset(Dataset):
    SSD = 'ssd'
    YOLO = 'yolo'

    class FileLocations:
        def __init__(self, indices_path, coordinates_path):
            self.indices_path = indices_path
            self.coordinates_path = coordinates_path

    def __init__(self, root, file_locations: FileLocations, type, resize_to=None, is_containing=True, seed=1,
                 transform=None, loader=default_loader):
        if type == self.SSD or type == self.YOLO:
            self.type = type
        else:
            raise ValueError('Invalid type is specified')

        self.root = root
        self.file_locations = file_locations
        self.is_containing = is_containing

        self.data = self._load_data(seed)

        self.resize_to = resize_to if resize_to else (416, 416)

        self.transform = transform
        self.loader = loader

    def _load_data(self, seed=1):
        coordinates = json.load(open(self.file_locations.coordinates_path, 'r'))['coordinates']
        coordinates = DataFrame.from_records(
            coordinates, columns=['class', 'shelf', 'frame', 'xleft', 'yupper', 'xright', 'ylower'])

        if self.is_containing:
            data = self._load_containing_frames(coordinates)
        else:
            data = self._load_non_containing_frames(seed)

        return data

    def _load_non_containing_frames(self, seed):
        data = []
        indices = json.load(open(self.file_locations.indices_path, 'r'))['without']
        Random(seed).shuffle(indices)

        for filename in indices:
            data.append((f'{filename}.jpg', []))

        return data

    def _load_containing_frames(self, coordinates):
        data = []
        indices = json.load(open(self.file_locations.indices_path, 'r'))['data']

        for class_data in indices:
            for filename in class_data[1]:
                shelf_frame = filename.split('-')
                shelf = int(shelf_frame[0].split('_')[1])
                frame = int(shelf_frame[1].split('frame')[1])
                image_values = coordinates[
                    (coordinates['shelf'] == shelf) & (coordinates['frame'] == frame)].values.tolist()
                for i, val in enumerate(image_values):
                    image_values[i] = [image_values[i][0], *image_values[i][3:]]
                data.append((f'{filename}.jpg', image_values))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, targets = self.data[index]
        image = self.loader(os.path.join(self.root, *filename.split('-')))

        if self.transform:
            image = self.transform(image)

        if self.type == self.SSD:
            targets = self._format_ssd_targets(targets)
            return (image, *targets)
        else:
            targets = self._format_yolo_targets(targets)
            return image, targets

    def _format_ssd_targets(self, targets):
        labels = [t[0] for t in targets]
        boxes = [t[1:] for t in targets]

        resized_width, resized_height = self.resize_to
        width_scale, height_scale = 720 / resized_width, 480 / resized_height
        new_boxes = [[box[0] / width_scale, box[1] / height_scale, box[2] / width_scale, box[3] / height_scale]
                     for box in boxes]

        normalized_boxes = [[box[0] / resized_width, box[1] / resized_height, box[2] / resized_width,
                             box[3] / resized_height] for box in new_boxes]

        labels = LongTensor(labels)
        normalized_boxes = FloatTensor(normalized_boxes)

        return normalized_boxes, labels

    def _format_yolo_targets(self, targets):
        resized_width, resized_height = self.resize_to
        width_scale, height_scale = 720 / resized_width, 480 / resized_height
        targets = [[box[0], box[1] / width_scale, box[2] / height_scale, box[3] / width_scale, box[4] / height_scale]
                   for box in targets]

        return targets


class EvalGroZiDetectionDataset(ImageFolder):
    SSD: str = 'SSD'
    YOLO: str = 'YOLO'

    def __init__(self, root, model: str = SSD, transform=Compose([]), combined_transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, transform_to_tensor: bool = True, min_resize: int = None,
                 max_resize: int = None, max_data_usage: int = 1, max_data_in_combined: int = 1, max_object: int =
                 None, seen_images: int = 0, batch_size: int = None, do_flip_augment: bool = True,
                 yolo_fixed_image_size: tuple = None, background_noise = None):
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
            self.yolo_fixed_image_size = yolo_fixed_image_size

        self.do_flip_augment = do_flip_augment
        self.max_data_usage = max_data_usage
        self.max_data_in_combined = max_data_in_combined
        self.kwargs = {
            'min_resize': min_resize,
            'max_resize': max_resize,
            'individual_transform': self.transform,
            'give_noise': background_noise
        }

        self.combined_transform = combined_transform
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
            path = sample[0]
            label = int(path.split(os.sep)[-2]) - 1
            images.append(self.loader(path))
            # index is added since bounding_boxes keys will have overlapping values due to same label being in the
            # same image if index is not added. By grouping index and label to a tuple, a unique key is guaranteed
            # and no overlapping value in bounding_boxes dictionary is possible even with products of the same label
            # occurring more than once in the combined_image.
            index_labels.append((index, label))
            self.usage[sample] += 1

        combined_image, bounding_boxes = combined_zoom_out(identifiers=index_labels, images=images, **self.kwargs)

        labels = [label for _, label in bounding_boxes.keys()]
        bounding_boxes = list(bounding_boxes.values())

        target = None
        if self.model == self.SSD:
            combined_image, *target = self._format_ssd(combined_image, labels, bounding_boxes)
        elif self.model == self.YOLO:
            combined_image, target = self._format_yolo(combined_image, labels, bounding_boxes)

        if self.combined_transform:
            combined_image = self.combined_transform(combined_image)

        if self.transform_to_tensor:
            combined_image = ToTensor()(combined_image)

        if self.model == self.SSD:
            return (combined_image, *target)
        elif self.model == self.YOLO:
            return combined_image, target

    def _format_yolo(self, combined_image, labels, bounding_boxes):
        assert len(labels) == len(bounding_boxes)

        yolo_labels = [[int(labels[i]), *bounding_boxes[i]] for i in range(len(labels))]

        shape = self.yolo_fixed_image_size if self.yolo_fixed_image_size else self._get_shape_for_multi_scale_training()

        width_scale, height_scale = 720 / shape[0], 480 / shape[1]
        targets = [[box[0], box[1] / width_scale, box[2] / height_scale, box[3] / width_scale, box[4] / height_scale]
                   for box in yolo_labels]

        return combined_image, targets

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

    def _format_ssd(self, combined_image, labels, bounding_boxes):
        labels = LongTensor(labels)
        bounding_boxes = FloatTensor(bounding_boxes)

        return ssd_transform(combined_image, bounding_boxes, labels, do_flip=self.do_flip_augment)

    def __get_used_samples(self):
        indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
        used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
        fully_used = [self.usage[k] == self.max_data_usage for k in used_samples]

        while all(fully_used):
            indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
            used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
            fully_used = [self.usage[k] == self.max_data_usage for k in used_samples]

        return used_samples

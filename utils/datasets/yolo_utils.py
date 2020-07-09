from random import randint

from torch.utils.data import ConcatDataset
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import Compose, Resize, ToTensor, RandomResizedCrop, ColorJitter, RandomHorizontalFlip, \
    RandomVerticalFlip


class UsageBasedDataset(ImageFolder):
    """A custom dataset for automatic balancing dataset based on individual item usages.

    While balancing dataset can be achieved in ``DataLoader`` using ``WeightedRandomSampler``, this dataset is
    proposed to **enforce** a specified item usage count for each class' data in the dataset. This enables automatic
    on-the-fly balancing mechanism in a ``Dataset`` instance itself, and requires only 1 additional parameter
    (``usage``) specifying the item usage count to be used.
    """

    def __init__(self, root, usage, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        self.usage = usage
        self.usages = {int(c) - 1: 0 for c in self.classes}

    def __len__(self):
        return self.usage * len(self.classes)

    def __getitem__(self, index):
        index = index if index < len(self.samples) else randint(0, len(self.samples) - 1)
        path, target = self.samples[index]

        while self.usages[target] >= self.usage:
            under_usage = [s for s in self.samples if self.usages[s[1]] < self.usage]
            index = randint(0, len(under_usage) - 1) if len(under_usage) > 1 else 0
            path, target = under_usage[index]
        self.usages[target] += 1

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_recognition_transform(resolution):
    return Compose([
        RandomResizedCrop(size=(resolution, resolution), scale=(.5, 1.)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(p=.05),
        ColorJitter(brightness=.25),
        ColorJitter(contrast=.25),
        ColorJitter(saturation=.25),
        ColorJitter(hue=.25),
        Resize((resolution, resolution)),
        ToTensor()
    ])


def get_recognition_dataset(item_source, resolution, increase_by_factor, item_usage):
    if item_usage:
        dataset = UsageBasedDataset(root=item_source, usage=item_usage, transform=get_recognition_transform(resolution))
    else:
        dataset = ImageFolder(root=item_source, transform=get_recognition_transform(resolution))

    if increase_by_factor:
        dataset = ConcatDataset([dataset for _ in range(increase_by_factor)])

    return dataset

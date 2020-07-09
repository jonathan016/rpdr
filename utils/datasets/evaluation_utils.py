import json
import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class RecognitionDataset(Dataset):
    VAL = 'val'
    TEST = 'test'

    def __init__(self, root, indices_path, files_path, type, transform=None, target_transform=None,
                 loader=default_loader):
        if type == self.VAL or type == self.TEST:
            self.type = type
        else:
            raise ValueError('Invalid type is specified')

        self.root = root
        self.indices_path = indices_path
        self.indices = self._load_indices(indices_path)
        self.files = json.load(open(files_path, 'r'))['classes_files']

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _load_indices(self, indices_path):
        indices = json.load(open(indices_path, 'r'))
        ret = []

        for k in indices.keys():
            ret.extend(indices[k][self.type])

        return ret

    def val(self):
        self.type = self.VAL
        self.indices = self._load_indices(self.indices_path)

    def test(self):
        self.type = self.TEST
        self.indices = self._load_indices(self.indices_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        target, path = self.files[index]
        path = os.path.join(self.root, str(target + 1), path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

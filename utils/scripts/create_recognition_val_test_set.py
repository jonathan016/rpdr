import json
import os
from argparse import ArgumentParser
from random import randint

inSituCrop_dir = '../../../../Dataset/[GR120] GroZi-120 Dataset/[GR120] In Situ Images/in_situ_jpgs'


def read_classes(dir):
    classes_files = []

    for file in os.listdir(os.path.join(inSituCrop_dir, dir)):
        classes_files.append((int(dir) - 1, file))

    return classes_files


def load_in_situ_crop():
    dirs = list(map(int, os.listdir(inSituCrop_dir)))
    classes_files = []

    for dir in dirs:
        classes_files.extend(read_classes(str(dir)))

    return classes_files


def test_for_class(cls, classes_files):
    indices_with = set()

    while len(indices_with) != 10:
        filtered = []
        for i, x in enumerate(classes_files):
            if int(x[0]) == int(cls) and i not in indices_with:
                filtered.append((i, x))
        if len(filtered) == 0:
            break
        filtered_potential_index = randint(0, len(filtered) - 1)
        potential_index = filtered[filtered_potential_index][0]
        if int(classes_files[potential_index][0]) == int(cls):
            indices_with.add(potential_index)

    return list(indices_with)


def val_for_class(cls, coordinates, test_with):
    indices_with = set()

    while len(indices_with) != 10:
        filtered = []
        for i, x in enumerate(coordinates):
            if int(x[0]) == int(cls) and i not in indices_with and i not in test_with:
                filtered.append((i, x))
        if len(filtered) == 0:
            break
        filtered_potential_index = randint(0, len(filtered) - 1)
        potential_index = filtered[filtered_potential_index][0]
        if int(coordinates[potential_index][0]) == int(cls) and potential_index not in test_with:
            indices_with.add(potential_index)

    return list(indices_with)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--in_situ_directory', default=None,
                        help='Path to in situ directory for recognition validation and testing')

    args = parser.parse_args()

    if args.in_situ_directory:
        inSituCrop_dir = args.in_situ_directory

    c = load_in_situ_crop()

    indices = {}

    for k in range(120):
        test_with = test_for_class(k, c)
        val_with = val_for_class(k, c, test_with)

        indices[k] = {}
        indices[k]['test'] = test_with
        indices[k]['val'] = val_with
        print('Finished processing class {}'.format(k))

    json.dump(indices, open('../../val_test/recog_val_test.json', 'w'))

    c = {'classes_files': c}
    json.dump(c, open('../../val_test/recog_val_test_classes_files.json', 'w'))

    data = json.load(open('../../val_test/recog_val_test.json', 'r'))

    test_max_with = 0
    test_min_with = 1000
    test_max_with_k = None
    test_min_with_k = None
    val_max_with = 0
    val_min_with = 1000
    val_max_with_k = None
    val_min_with_k = None

    for k in data.keys():
        length = len(data[k]['test'])
        if length > test_max_with:
            test_max_with = length
            test_max_with_k = k
        if length < test_min_with:
            test_min_with = length
            test_min_with_k = k

        length = len(data[k]['val'])
        if length > val_max_with:
            val_max_with = length
            val_max_with_k = k
        if length < val_min_with:
            val_min_with = length
            val_min_with_k = k

    print(test_max_with)
    print(test_min_with)
    print(test_max_with_k)
    print(test_min_with_k)
    print(val_max_with)
    print(val_min_with)
    print(val_max_with_k)
    print(val_min_with_k)

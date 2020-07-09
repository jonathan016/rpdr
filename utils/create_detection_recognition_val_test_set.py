import os
import json
import pandas as pd
from random import randint

from argparse import ArgumentParser

inSituCrop_dir = '../../../Dataset/[GR120] GroZi-120 Dataset/[GR120] In Situ Images/inSitu'


def read_coordinates(dir):
    coordinates = []

    with open(os.path.join(inSituCrop_dir, dir, 'coordinates.txt')) as file:
        df = pd.read_csv(file, sep='\t', header=None)
        for i in range(len(df)):
            row = df.loc[i]
            coord = (int(dir) - 1, int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[2] + row[4]),
                     int(row[3] + row[5]))
            coordinates.append(coord)

    return coordinates


def load_in_situ_crop():
    dirs = list(map(int, os.listdir(inSituCrop_dir)))
    coordinates = []

    for dir in dirs:
        coordinates.extend(read_coordinates(str(dir)))

    return coordinates


def test_for_class(cls, coordinates):
    indices_without = set()
    indices_with = set()

    while len(indices_without) != 100:
        potential_index = randint(0, len(coordinates) - 1)
        if coordinates[potential_index][0] != cls:
            indices_without.add(potential_index)

    while len(indices_with) != 10:
        filtered = []
        for i, x in enumerate(coordinates):
            if int(x[0]) == int(cls) and i not in indices_with:
                filtered.append((i, x))
        if len(filtered) == 0:
            break
        filtered_potential_index = randint(0, len(filtered) - 1)
        potential_index = filtered[filtered_potential_index][0]
        if int(coordinates[potential_index][0]) == int(cls) and potential_index not in indices_without:
            indices_with.add(potential_index)

    return list(indices_without), list(indices_with)


def val_for_class(cls, coordinates, test_without, test_with):
    indices_without = set()
    indices_with = set()

    while len(indices_without) != 100:
        potential_index = randint(0, len(coordinates) - 1)
        if coordinates[potential_index][0] != cls and potential_index not in test_without:
            indices_without.add(potential_index)

    while len(indices_with) != 10:
        filtered = []
        for i, x in enumerate(coordinates):
            if int(x[0]) == int(cls) and i not in indices_with and i not in test_with:
                filtered.append((i, x))
        if len(filtered) == 0:
            break
        filtered_potential_index = randint(0, len(filtered) - 1)
        potential_index = filtered[filtered_potential_index][0]
        if int(coordinates[potential_index][0]) == int(cls) and potential_index not in indices_without and \
                potential_index not in test_with:
            indices_with.add(potential_index)

    return list(indices_without), list(indices_with)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--in_situ_directory', default=None,
                        help='Path to in situ directory with coordinate annotations')

    args = parser.parse_args()

    if args.in_situ_directory:
        inSituCrop_dir = args.in_situ_directory

    c = load_in_situ_crop()

    indices = {}

    for k in range(120):
        test_without, test_with = test_for_class(k, c)
        val_without, val_with = val_for_class(k, c, test_without, test_with)

        indices[k] = {}
        indices[k]['test'] = {}
        indices[k]['test']['without'] = test_without
        indices[k]['test']['with'] = test_with
        indices[k]['val'] = {}
        indices[k]['val']['without'] = val_without
        indices[k]['val']['with'] = val_with
        print('Finished processing class {}'.format(k))

    json.dump(indices, open('../val_test/val_test.json', 'w'))

    c = {'coordinates': c}
    json.dump(c, open('../val_test/val_test_coordinates.json', 'w'))

    data = json.load(open('../val_test/val_test.json', 'r'))

    test_max_without = 0
    test_min_without = 1000
    test_max_without_k = None
    test_min_without_k = None
    test_max_with = 0
    test_min_with = 1000
    test_max_with_k = None
    test_min_with_k = None
    val_max_without = 0
    val_min_without = 1000
    val_max_without_k = None
    val_min_without_k = None
    val_max_with = 0
    val_min_with = 1000
    val_max_with_k = None
    val_min_with_k = None

    for k in data.keys():
        length = len(data[k]['test']['without'])
        if length > test_max_without:
            test_max_without = length
            test_max_without_k = k
        if length < test_min_without:
            test_min_without = length
            test_min_without_k = k

        length = len(data[k]['val']['without'])
        if length > val_max_without:
            val_max_without = length
            val_max_without_k = k
        if length < val_min_without:
            val_min_without = length
            val_min_without_k = k

        length = len(data[k]['test']['with'])
        if length > test_max_with:
            test_max_with = length
            test_max_with_k = k
        if length < test_min_with:
            test_min_with = length
            test_min_with_k = k

        length = len(data[k]['val']['with'])
        if length > val_max_with:
            val_max_with = length
            val_max_with_k = k
        if length < val_min_with:
            val_min_with = length
            val_min_with_k = k

    print(test_max_without)
    print(test_min_without)
    print(test_max_without_k)
    print(test_min_without_k)
    print(test_max_with)
    print(test_min_with)
    print(test_max_with_k)
    print(test_min_with_k)
    print(val_max_without)
    print(val_min_without)
    print(val_max_without_k)
    print(val_min_without_k)
    print(val_max_with)
    print(val_min_with)
    print(val_max_with_k)
    print(val_min_with_k)

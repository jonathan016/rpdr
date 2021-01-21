import json
import os
from copy import deepcopy
from random import randrange

from pandas import DataFrame, read_csv as pd_read_csv


def read_coordinates_of_class(in_situ_crop_directory, directory_as_class):
    class_coordinates = []

    with open(os.path.join(in_situ_crop_directory, directory_as_class, 'coordinates.txt')) as file:
        df = pd_read_csv(file, sep='\t', header=None)
        for i in range(len(df)):
            row = df.loc[i]

            cls = int(directory_as_class) - 1
            shelf = int(row[0])
            frame = int(row[1])
            xleft = int(row[2])
            yupper = int(row[3])
            xright = int(row[2] + row[4])
            ylower = int(row[3] + row[5])

            class_coordinates.append((cls, shelf, frame, xleft, yupper, xright, ylower))

    return class_coordinates


def load_coordinates_from_in_situ_crop(in_situ_crop_directory):
    dirs = list(map(int, os.listdir(in_situ_crop_directory)))
    all_coordinates = []

    for directory in dirs:
        all_coordinates.extend(read_coordinates_of_class(in_situ_crop_directory, str(directory)))

    return all_coordinates


def load_save_and_filter_coordinates(coordinates_save_file, in_situ_crop_directory):
    loaded_coordinates = load_coordinates_from_in_situ_crop(in_situ_crop_directory)

    json.dump({'coordinates': loaded_coordinates}, open(coordinates_save_file, 'w'))

    loaded_coordinates = DataFrame.from_records(
        loaded_coordinates, columns=['class', 'shelf', 'frame', 'xleft', 'yupper', 'xright', 'ylower'])
    loaded_coordinates = loaded_coordinates[
        loaded_coordinates['frame'] >= 0]  # Remove frames with value < 0 (2 frames with this condition)

    # Remove coordinates of shelf 16 and above frame 4318 as the video is corrupted and frames above 4318 is unavailable
    # Classes 20, 22, and 92 will have 0 containing frames as these classes are available in the corrupted frames
    # Note that indexing of classes starts from 0
    corrupted_frames = loaded_coordinates[(loaded_coordinates['shelf'] == 16) & (loaded_coordinates['frame'] > 4318)]
    loaded_coordinates = loaded_coordinates.drop([row[0] for row in corrupted_frames.iterrows()])

    return loaded_coordinates


def get_frames_without_any_object(all_coordinates, interval):
    frames_without_any_object = []

    for shelf in range(1, max(all_coordinates['shelf']) + 1):
        directory_contents = os.listdir(
            f'../../../../Dataset/[GR120] GroZi-120 Dataset/[GR120] In Situ Videos/video/Shelf_{shelf}')
        directory_frames = [x for x in directory_contents if x.startswith('frame') and x.endswith('.jpg')]
        total_frames = len(directory_frames)

        frames = all_coordinates[all_coordinates['shelf'] == shelf]
        frames = frames.sort_values(['frame'])

        for frame in range(total_frames):
            annotated_frames = frames[(frames['frame'] >= frame - interval) & (frames['frame'] <= frame + interval)]
            if annotated_frames.empty:
                frames_without_any_object.append(f'Shelf_{shelf}-frame{frame}')

    return frames_without_any_object


def get_test_frames_by_step(frames_without_any_object, step):
    selected_test_frames = []

    while len(selected_test_frames) != 12000:
        selected = frames_without_any_object[::step]
        if len(selected) + len(selected_test_frames) > 12000:
            selected = selected[:12000 - len(selected_test_frames)]
        selected_test_frames.extend(selected)
        for s in selected:
            frames_without_any_object.remove(s)

    return selected_test_frames


def get_validation_frames_by_step(frames_without_any_object, step):
    selected_validation_frames = []

    if len(frames_without_any_object) <= 12000:
        selected_validation_frames = frames_without_any_object
    else:
        while len(selected_validation_frames) <= 12000 and len(frames_without_any_object) != 0:
            selected = frames_without_any_object[::step]
            if len(selected) + len(selected_validation_frames) > 12000:
                selected = selected[:12000 - len(selected_validation_frames)]
            selected_validation_frames.extend(selected)
            for s in selected:
                frames_without_any_object.remove(s)

    return selected_validation_frames


def select_non_containing_frames(all_coordinates, interval, step):
    frames_without_any_object = get_frames_without_any_object(all_coordinates, interval)
    print(f'Non containing frames count: {len(frames_without_any_object)}')

    # Select frames by step
    selected_test_frames = get_test_frames_by_step(frames_without_any_object, step)
    print(f'Selected non containing test frames count: {len(selected_test_frames)}')
    selected_validation_frames = get_validation_frames_by_step(frames_without_any_object, step)
    print(f'Selected non containing validation frames count: {len(selected_validation_frames)}')

    # Construct variables for output files
    test_frames_without = {'without': selected_test_frames}
    validation_frames_without = {'without': selected_validation_frames}

    # Output to file
    json.dump(test_frames_without, open(f'../../val_test/detect_test_frames-not_containing.json', 'w'))
    json.dump(validation_frames_without, open(f'../../val_test/detect_val_frames-not_containing.json', 'w'))


def extract_classes_in_frames(all_coordinates):
    frames_labels = {}
    frame_occurrences = {}

    for _, coordinate in all_coordinates.iterrows():
        # Fill frames_labels data
        key, value = f'Shelf_{coordinate["shelf"]}-frame{coordinate["frame"]}', coordinate['class']
        if key in frames_labels:
            frames_labels[key].append(value)
        else:
            frames_labels[key] = [value]

        # Fill frame_occurrences data
        key, value = reversed((key, value))
        if key in frame_occurrences:
            frame_occurrences[key].add(value)
        else:
            frame_occurrences[key] = {value}

    for k in frames_labels.keys():
        frames_labels[k] = sorted(frames_labels[k])

    return frame_occurrences, frames_labels


def random_select_balanced(present_classes, target_class, expected_balance=10, trials=100, is_validation=False):
    contains_target_class = [{target_class}.issubset(classes) for classes in present_classes]

    current_balance = 0
    returned = deepcopy(contains_target_class)
    indices = [i for i in range(len(present_classes)) if contains_target_class[i]]

    selected_indices = set()
    for i in range(trials):
        while len(indices) != 0:
            if current_balance == expected_balance:
                break
            index = randrange(0, len(indices))
            classes = present_classes[indices[index]]
            filtered = list(filter(lambda x: x == target_class, classes))
            current_balance += len(filtered)
            selected_indices.add(indices[index])
            if current_balance > expected_balance:
                returned[indices[index]] = False
                selected_indices.remove(indices[index])
            indices.pop(index)
        if current_balance == expected_balance:
            break
        else:
            current_balance = 0
            returned = deepcopy(contains_target_class)
            indices = [i for i in range(len(present_classes)) if contains_target_class[i]]

    indices = [i for i in range(len(present_classes)) if contains_target_class[i]]
    max_item = 1
    while len(selected_indices) < 10:
        if is_validation and len(indices) == 0:
            break
        index = randrange(0, len(indices))
        if sum([int(len([x for x in present_classes[i] if x == target_class]) == max_item) for i in indices]) == 0:
            max_item += 1
        if len([x for x in present_classes[indices[index]] if x == target_class]) == max_item:
            selected_indices.add(indices[index])
            indices.pop(index)

    for i in range(len(returned)):
        returned[i] = i in selected_indices

    return returned


def get_containing_frames(classes_sorted_by_least_occurrence, eval_set, validation_set):
    test_frames = {k: set() for k in range(120)}
    validation_frames = {k: set() for k in range(120)}

    corrupted_classes = [20, 22, 92]
    while any([(False if i in corrupted_classes else len(v) != 10) for i, v in enumerate(test_frames.values())]):
        for c in classes_sorted_by_least_occurrence:
            if len(test_frames[c]) != 10:
                potential_frames = eval_set[random_select_balanced(list(eval_set['present_classes']), c)]  # Filter
                test_frames[c] = set(potential_frames['shelf_frame'])

                # Ensure no test data is used in validation set
                validation_set = validation_set.drop([row[0] for row in potential_frames.iterrows()], errors='ignore')

    for c in classes_sorted_by_least_occurrence:
        potential_frames = validation_set[
            random_select_balanced(list(validation_set['present_classes']), c, is_validation=True)]
        validation_frames[c] = set(potential_frames['shelf_frame'])
        if len(validation_frames[c]) != 0:
            assert not validation_frames[c].issubset(test_frames[c])

    print(f'Containing test frames count: {sum([len(v) for v in test_frames.values()])}')
    print(f'Containing validation frames count: {sum([len(v) for v in validation_frames.values()])}')

    return test_frames, validation_frames


def select_containing_frames(all_coordinates):
    all_coordinates = all_coordinates.sort_values(['shelf', 'frame', 'class'])

    frame_occurrences, frames_labels = extract_classes_in_frames(all_coordinates)

    eval_set = DataFrame.from_records(
        [[k, v] for k, v in frames_labels.items()], columns=['shelf_frame', 'present_classes'])
    validation_set = eval_set.copy(True)
    classes_sorted_by_least_occurrence = DataFrame.from_records(
        [[k, len(v)] for k, v in frame_occurrences.items()], columns=['class', 'total_frames']).sort_values(
        ['total_frames'])['class'].values

    test_frames, validation_frames = get_containing_frames(classes_sorted_by_least_occurrence, eval_set, validation_set)

    test_set = DataFrame.from_records([[k, list(v)] for k, v in test_frames.items()], columns=['class', 'frames'])
    test_set.to_json('../../val_test/detect_test_frames-containing.json', orient='split', index=False)
    validation_set = DataFrame.from_records([[k, list(v)] for k, v in validation_frames.items()],
                                            columns=['class', 'frames'])
    validation_set.to_json('../../val_test/detect_val_frames-containing.json', orient='split', index=False)


if __name__ == '__main__':
    in_situ_crop_dir = '../../../../Dataset/[GR120] GroZi-120 Dataset/[GR120] In Situ Images/inSitu'
    coordinates = load_save_and_filter_coordinates('../../val_test/detect_val_test_coordinates.json', in_situ_crop_dir)

    select_non_containing_frames(coordinates, interval=65, step=5)
    select_containing_frames(coordinates)

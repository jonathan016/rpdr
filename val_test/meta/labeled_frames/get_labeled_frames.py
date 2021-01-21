import json

from pandas import DataFrame

if __name__ == '__main__':
    # Load frames with object data
    coordinates = json.load(open('val_test_coordinates.json', 'r'))['coordinates']

    df = DataFrame.from_records(
        coordinates, columns=['class', 'shelf', 'frame', 'xleft', 'yupper', 'xright', 'ylower'])
    df = df[df['frame'] >= 0]  # Remove frames with value below 0 (2 frames exist with this condition)
    df = df.sort_values(['shelf', 'frame', 'class'])

    # Fill data based on annotation occurrence
    frames_labels = {}
    frame_occurrences = {}
    class_annotation = {}
    for _, row in df.iterrows():
        # Fill frames_labels data
        key = f'Shelf{row["shelf"]}_frame{row["frame"]}'
        if key in frames_labels:
            frames_labels[key].append(row['class'])
        else:
            frames_labels[key] = [row['class']]

        # Fill class_annotation data
        key = row['class']
        if key in class_annotation:
            class_annotation[key] += 1
        else:
            class_annotation[key] = 1

        # Fill frame_occurrences data
        key = row['class']
        value = f'Shelf{row["shelf"]}_frame{row["frame"]}'
        if key in frame_occurrences:
            frame_occurrences[key].add(value)
        else:
            frame_occurrences[key] = {value}

    # Save frame-label data
    fl = DataFrame.from_records([[k, v, len(v)] for k, v in frames_labels.items()],
                                columns=['shelf_frame', 'present_classes', 'annotated_objects'])
    fl.to_csv('with_labels.csv', index=False)

    # Save frame-class-label data
    ca = DataFrame.from_records([[k, len(v), v] for k, v in frame_occurrences.items()],
                                columns=['class', 'total_frames', 'frames'])
    ca.to_csv('class_occurrence.csv', index=False)

    # Save class-annotation data
    cod = DataFrame.from_records([[k, v] for k, v in class_annotation.items()], columns=['class', 'annotations'])
    cod.to_csv('class_annotations.csv', index=False)

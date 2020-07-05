import argparse
import numpy as np
import os
import random

from PIL import Image

"""Created by @AlexeyAB and modified by @mhaghighat to generate anchor boxes ratio for YOLO models.

Some modifications are made. See https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py for more 
details.
"""

width_in_cfg_file = 416.
height_in_cfg_file = 416.


def IOU(x, centroids):
    similarities = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file, yolo_version):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file / (32. if yolo_version == 2 else 1.)
        anchors[i][1] *= height_in_cfg_file / (32. if yolo_version == 2 else 1.)

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, anchor_file, yolo_version):
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file, yolo_version)
            return

        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def get_from_folder(folder, normalize):
    annotation_dims = []

    if normalize != 0:
        max_w = 0.
        max_h = 0.
        for dir in os.listdir(folder):
            for img in os.listdir(os.path.join(folder, dir)):
                w, h = Image.open(os.path.join(folder, dir, img)).size
                max_w = w if w > max_w else max_w
                max_h = h if h > max_h else max_h
    else:
        max_w = 1.
        max_h = 1.

    for dir in os.listdir(folder):
        for img in os.listdir(os.path.join(folder, dir)):
            w, h = Image.open(os.path.join(folder, dir, img)).size
            annotation_dims.append(tuple(map(float, (w / max_w, h / max_h))))
    annotation_dims = np.array(annotation_dims)

    return annotation_dims


def cluster(annotation_dims, args, num_clusters):
    anchor_file = os.path.join(args.output_dir, 'anchors%d.txt' % (num_clusters))

    indices = [random.randrange(annotation_dims.shape[0]) for _ in range(num_clusters)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims, centroids, anchor_file, args.yolo_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default='./', type=str, help='Source directory of data\n')
    parser.add_argument('-o', '--output_dir', default='./', type=str, help='Output anchor directory\n')
    parser.add_argument('-nc', '--num_clusters', default=0, type=int, help='Number of clusters\n')
    parser.add_argument('-n', '--normalize', default=0, type=int,
                        help='1 to normalize by largest width and height, 0 otherwise\n')
    parser.add_argument('-yv', '--yolo_version', default=2, type=int, help='Version of YOLO\n')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    annotation_dims = get_from_folder(args.source, args.normalize)

    if args.num_clusters == 0:
        for num_clusters in range(1, 11):
            cluster(annotation_dims, args, num_clusters)
    else:
        cluster(annotation_dims, args, args.num_clusters)

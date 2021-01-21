import os
from argparse import ArgumentParser

import cv2

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', default='~/grozi_zip/', help='Root of video to be extracted')
    parser.add_argument('-o', default='~/rpdr-config-results/data/frames/', help='Frames output containing folder')

    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.mkdir(args.o)

    videos = ['Shelf_1', 'Shelf_2', 'Shelf_3', 'Shelf_4', 'Shelf_5', 'Shelf_6', 'Shelf_7', 'Shelf_8', 'Shelf_9',
              'Shelf_10', 'Shelf_11', 'Shelf_12', 'Shelf_13', 'Shelf_14', 'Shelf_15', 'Shelf_16', 'Shelf_17',
              'Shelf_18', 'Shelf_19', 'Shelf_20', 'Shelf_21', 'Shelf_22', 'Shelf_23', 'Shelf_24', 'Shelf_25',
              'Shelf_26', 'Shelf_27', 'Shelf_28', 'Shelf_29']

    for video in videos:
        os.mkdir(os.path.join(args.o, video))
        vidcap = cv2.VideoCapture(os.path.join(args.r, video + '.avi'))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(args.o, video, 'frame' + str(count) + '.jpg'), image)
            success, image = vidcap.read()
            count += 1
        print('Success processed video #' + video)

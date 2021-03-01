"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2
import argparse
from glob import glob

from src.visualize import vis_utils as vis

from src.io.psee_loader import PSEELoader


def play_files_parallel(td_files, labels=None, delta_t=80000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    # use the naming pattern to find the corresponding box file
    box_videos1 = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0]) for td_file in td_files]
    box_videos2 = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*_result.npy')[0]) for td_file in td_files]
    
    height, width = videos[0].get_size()
    labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

    # optionally skip n minutes in all videos
    for v in videos + box_videos1:
        v.seek_time(skip)

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    last_boxes = np.empty((0,))
    ts = 0
    ts_last_box = 0

    # while all videos have something to read
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]
        box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos1]
        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)

            if len(evs) != 0:
                ts = evs[0][0]
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)
            vis.draw_bboxes(im, boxes, labelmap=labelmap)

            if len(boxes) == 0:
                vis.draw_bboxes(im, last_boxes, labelmap=labelmap)
            else:
                last_boxes = boxes.copy()
                ts_last_box = ts
            if ts - ts_last_box > 200000:
                last_boxes = np.empty((0,))


        # display the result
        cv2.imshow('out', frame)
        cv2.waitKey(1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+",
                        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=0, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=10000, type=int, help="load files by delta_t in microseconds")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, skip=ARGS.skip, delta_t=ARGS.delta_t)

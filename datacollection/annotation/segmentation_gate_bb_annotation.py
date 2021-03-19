import cv2
import os
import json
import shutil
import numpy as np
from shutil import copyfile
import json
import math

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 
from mydataloader.prophesee import npy_events_tools

from quaternion_utils import Quaternion



class GateBoundingBoxAnnotation:
    """ Annotate gates for training using back projection

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):

        self.resolution = [240, 180]
        self.trackid_counter = 0

    def computeBoundingBox(self, seg_image):

        start_x, start_y = -1, -1
        end_x, end_y = -1, -1

        for y in range(self.resolution[1]):
            if np.sum(seg_image[y, :]) > 0.0:
                end_y = y

                if start_y == -1:
                    start_y = y

        for x in range(self.resolution[0]):
            if np.sum(seg_image[:, x]) > 0.0:
                end_x = x

                if start_x == -1:
                    start_x = x

        x = start_x
        y = start_y
        w = end_x - start_x
        h = end_y - start_y
        class_id = 1
        confidence = 1

        return x, y, w, h, class_id, confidence


    def run(self, source_folder, dest_folder, lookup=None):

        print("[*] Source folder: %s" % source_folder)
        output_dtype = np.dtype([('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        
        # Get all attribute files
        segFiles = []
        for file in os.listdir(source_folder):
            if file.endswith(".npy"):
                segFiles.append(file)
                  

        num_of_samples = len(segFiles)
        print("[*] Num of samples found: %d" % num_of_samples)

        # Create output folder.
        if not os.path.isdir(dest_folder):
            print("%s created." % dest_folder)
            os.mkdir(dest_folder)

        output = np.empty((0,), dtype = output_dtype)
        for file in segFiles:
            track_id = 0
            
            seg_images = np.load(os.path.join(source_folder, file), allow_pickle=True)
            for i, ts in enumerate(seg_images[:,0]):
                            
                x, y, w, h, class_id, confidence = self.computeBoundingBox(seg_images[i, 1])
                
                if x != -1:
                    bbox_out = tuple((ts, x, y, w, h, class_id, confidence, track_id))
    
                    temp_arr = np.array(bbox_out, dtype=output_dtype)
                    output = np.append(output, temp_arr)

                    track_id += 1
                        
            split_name = file.split('_')
            save_file = split_name[0] + '_' + split_name[1] + '_' + split_name[2] + '_bbox.npy'    
            np.save(os.path.join(dest_folder, save_file), output)
                    


def main():
    source_folder = os.path.join(parentdir, 'source_data')
    dest_folder = os.path.join(parentdir, 'dest_data')
    
    bb = GateBoundingBoxAnnotation()
    bb.run(source_folder, dest_folder)
    
if __name__ == "__main__":
    main()
import numpy as np
import airsim
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt
import argparse
import sys, signal
import pandas as pd
import pickle
from event_simulator import *

parser = argparse.ArgumentParser(description="Simulate event data from AirSim")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--width", type=int, default=304)


class AirSimEventGen:
    def __init__(self, W, H, debug=False):
        self.ev_sim = EventSimulator(W, H)
        self.W = W
        self.H = H

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )

        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.init = True
        self.start_ts = None

        self.rgb_image_shape = [H, W, 3]
        self.debug = debug

        self.event_fmt = "%1.7f", "%d", "%d", "%d"
        self.events = None

        if debug:
            self.fig, self.ax = plt.subplots(1, 1)

        ## Attribute collection
        self.counter = 0
        self.attrFrequency = 2 # Hz
        self.singleDroneAttribute = np.zeros(14)
        self.droneAttributes = np.zeros([14])


    def visualize_events(self, event_img):
        event_img = self.convert_event_img_rgb(event_img)
        self.ax.cla()
        self.ax.imshow(event_img, cmap="viridis")
        plt.draw()
        plt.pause(0.001)

    def convert_event_img_rgb(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip(image, 0, 1) * 255
        out[:, :, 2] = np.clip(image, -1, 0) * -255

        return out

    def save_to_files(self, tstart, tend):
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        np.save('events_' + date + '_' + str(int(tstart)) + '_' + str(int(tend)) + '.npy', self.events[1:])
        np.save('attributes_' + date + '_' + str(int(tstart)) + '_' + str(int(tend)) + '.npy', self.droneAttributes[:,1:])
        self.events = None
        self.droneAttributes = np.zeros([14])


    def _stop_event_gen(self, signal, frame):
        print("\nCtrl+C received. Stopping event sim...")
        sys.exit(0)

    ## Attribute collection
    # Attributes are collected in a .npy file with the following formatting:
    # [ts, pos_x, pos_y, pos_z, att_w, att_x, att_y, att_z, 
    # linvel_x, linvel_y, linvel_z, angvel_x, angvel_y, angvel_z]
    # We might assume that the gates are static and does not change position and pose
    def collectData(self, ts):
        self.singleDroneAttribute[0] = ts
        kinematics = self.client.simGetGroundTruthKinematics('PX4')
        pos = kinematics.position
        ori = kinematics.orientation
        linvel = kinematics.linear_velocity
        angvel = kinematics.angular_velocity


        self.singleDroneAttribute[1:8] = [pos.x_val, pos.y_val, pos.z_val,
                                          ori.w_val, ori.x_val, ori.y_val, ori.z_val]
        self.singleDroneAttribute[8:14] = [linvel.x_val, linvel.y_val, linvel.z_val,
                                           angvel.x_val, angvel.y_val, angvel.z_val]

        obj = self.client.simGetObjectPose('OrangeBall')

        self.droneAttributes = np.c_[self.droneAttributes, self.singleDroneAttribute]
        

    def saveAttrToNpy(self):
        np.save(self.attrFilename, self.droneAttributes)


if __name__ == "__main__":
    args = parser.parse_args()

    event_generator = AirSimEventGen(args.width, args.height, debug=args.debug)
    number_of_trials = 3

    signal.signal(signal.SIGINT, event_generator._stop_event_gen)

    for i in range(number_of_trials):
        t_start = time.time()
        print("here we go")
        while (time.time() - t_start) < 10:
            image_request = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            tnew = time.time_ns()
            response = event_generator.client.simGetImages([event_generator.image_request])
            while response[0].height == 0 or response[0].width == 0:
                response = event_generator.client.simGetImages(
                    [event_generator.image_request]
                )
            print("time grab: " + str((time.time_ns() - tnew)/1000000))
            ts = time.time_ns()

            if event_generator.init:
                event_generator.start_ts = ts
                event_generator.init = False

            img = np.reshape(
                np.fromstring(response[0].image_data_uint8, dtype=np.uint8),
                event_generator.rgb_image_shape,
            )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Add small number to avoid issues with log(I)
            img = cv2.add(img, 0.001)

            ts = time.time_ns()
            ts_delta = (ts - event_generator.start_ts) * 1e-3

            # Event sim keeps track of previous image automatically
            event_img, events = event_generator.ev_sim.image_callback(img, ts_delta)

            tnew = time.time_ns()
            if events is not None and events.shape[0] > 0:
                event_generator.events = np.r_[event_generator.events, events]
                event_generator.collectData(events[0][0])
                if event_generator.debug:
                    event_generator.visualize_events(event_img)
            print("time vis: " + str((time.time_ns() - tnew)/1000000))

        event_generator.save_to_files(t_start, time.time())



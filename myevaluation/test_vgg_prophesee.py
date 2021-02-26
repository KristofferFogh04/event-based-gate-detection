"""
Example command: python -m unittests.sparse_VGG_test
"""
import numpy as np
import torch
import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

import sys, os
sys.path.insert(0, 'rpg_asynet')

from myconfig.settings import Settings
from mydataloader.dataset import getDataloader
from mymodels.asyn_sparse_vgg import EvalAsynSparseVGGModel
from mymodels.facebook_sparse_object_det import FBSparseObjectDet
from mymodels.yolo_detection import yoloDetect
from mymodels.yolo_detection import nonMaxSuppression
from mytraining.trainer import AbstractTrainer
import rpg_asynet.utils.visualizations as visualizations
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
device = torch.device("cpu")

class TestSparseVGG():

    def __init__(self, args, settings, save_dir='log/PropheseeResults',):
        self.settings = settings
        self.save_dir = save_dir
        self.args = args
        self.multi_processing = args.use_multiprocessing
        self.compute_active_sites = args.compute_active_sites

        self.nr_classes = 2
        self.nr_input_channels = 2
        self.sequence_length = 60
        self.output_map = 6 * 8
        self.model_input_size = torch.tensor([self.settings.height, self.settings.width])
        self.nr_events_timestep = [self.settings.nr_events_window, self.settings.nr_events_window]

        self.writer = SummaryWriter(self.settings.ckpt_dir)

    def test_sparse_VGG(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        for i_test in tqdm.tqdm(range(1)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')

            # ---- Facebook VGG ----
            fb_model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection')).eval()
            spatial_dimensions = fb_model.spatial_size
            pth = 'log/trained-Prophesee/checkpoints/model_step_11.pth'
            fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

            print("initialized sync fb model")

            # ---- Asynchronous VGG ----
            layer_list =  [['C', self.nr_input_channels, 16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                           ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                           ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                           ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                           ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'],
                           ['ClassicC', 256, 256, 3, 2], ['ClassicBNRelu'],
                           ['ClassicFC', 256*self.output_map, 1024],  ['ClassicFC', 1024, self.output_map*(self.nr_classes + 5*self.nr_input_channels)]]

            asyn_model = EvalAsynSparseVGGModel(nr_classes=self.nr_classes, layer_list=layer_list, 
                                                device=device, input_channels=self.nr_input_channels)
            asyn_model.setWeightsEqual(fb_model)
            print("initialized asyn vgg model")

            # ---- Create Input -----
            event_window = 1000
            sequence_length = 10

            dataloader = getDataloader(self.settings.dataset_name)
            test_dataset = dataloader(self.settings.dataset_path, 'all', self.settings.height,
                                            self.settings.width, augmentation=False, mode='validation',
                                            nr_events_window=event_window, shuffle=True)
            self.object_classes = test_dataset.object_classes

            events, labels, histogram = test_dataset.__getitem__(0)

            # Histogram for synchronous network
            histogram = torch.from_numpy(histogram[np.newaxis, :, :])
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = AbstractTrainer.denseToSparse(histogram)


            list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
            input_histogram = torch.zeros(list_spatial_dimensions + [2])
            step_size = event_window // sequence_length

            # Detect using synchronous fb network on the whole batch
            fb_output = fb_model([locations, features])

            fb_detected_bbox = yoloDetect(fb_output, self.model_input_size.to(fb_output.device),
                   threshold=0.3)

            fb_detected_bbox = nonMaxSuppression(fb_detected_bbox, iou=0.6)
            fb_detected_bbox = fb_detected_bbox.long().cpu().numpy()


            with torch.no_grad():
                for i_sequence, nr_events in enumerate(self.nr_events_timestep):

                    #Generate input reprensetation for asynchrnonous network
                    new_batch_events = events[(step_size*i_sequence):(step_size*(i_sequence + 1)), :]
                    update_locations, new_histogram = asyn_model.generateAsynInput(new_batch_events, spatial_dimensions,
                                                                           original_shape=[self.settings.height, self.settings.width])
                    input_histogram = input_histogram + new_histogram
                    x_asyn = [None] * 5
                    x_asyn[0] = update_locations[:, :2].to(device)
                    x_asyn[1] = input_histogram.to(device)


                    # Detect using async network
                    asyn_output1 = asyn_model.forward(x_asyn)
                    asyn_output = asyn_output1[1].view([-1] + [6,8] + [(self.nr_classes + 5*self.nr_input_channels)])
                    asyn_detected_bbox = yoloDetect(asyn_output.float(), self.model_input_size.to(asyn_output.device),
                           threshold=0.3)
                    asyn_detected_bbox = nonMaxSuppression(asyn_detected_bbox, iou=0.6)
                    asyn_detected_bbox = asyn_detected_bbox.long().cpu().numpy()

            #print("FB: ")
            #print(fb_detected_bbox)
            #print("ASYN:")
            #print(asyn_detected_bbox)

            batch_one_mask = locations[:, -1] == 0
            vis_locations = locations[batch_one_mask, :2]
            features = features[batch_one_mask, :]
            vis_detected_bbox = fb_detected_bbox[fb_detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

            image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                      features=features.cpu().numpy())

            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                    class_name=[self.object_classes[i]
                                                                for i in fb_detected_bbox[:, -1]],
                                                     ground_truth=False, rescale_image=True)

            self.writer.add_image('FB', image, 1, dataformats='HWC')


            batch_one_mask = locations[:, -1] == 0
            vis_locations = locations[batch_one_mask, :2]
            features = features[batch_one_mask, :]
            vis_detected_bbox = asyn_detected_bbox[asyn_detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

            image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                      features=features.cpu().numpy())

            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                    class_name=[self.object_classes[i]
                                                                for i in asyn_detected_bbox[:, -1]],
                                                     ground_truth=False, rescale_image=True)

            self.writer.add_image('ASYN', image, 1, dataformats='HWC')

            file_path = os.path.join(self.settings.ckpt_dir, 'test_results.pth')
            torch.save({'state_dict': fb_model.state_dict()}, file_path)



def main():
    parser = argparse.ArgumentParser(description='Test network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--save_dir', help='Path to save location', required=True)
    parser.add_argument('--representation', default="")
    parser.add_argument('--use_multiprocessing', help='If multiprocessing should be used', action='store_true')
    parser.add_argument('--compute_active_sites', help='If active sites should be calculated', action='store_true')

    args = parser.parse_args()
    settings_filepath = args.settings_file
    save_dir = args.save_dir

    settings = Settings(settings_filepath, generate_log=False)

    tester = TestSparseVGG(args, settings)
    tester.test_sparse_VGG()


if __name__ == "__main__":
    main()

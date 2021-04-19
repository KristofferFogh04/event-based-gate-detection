import torch
import torch.nn as nn
import sparseconvnet as scn
from mylayers.conv_LSTM import ConvLSTM

class customREDnetSparseObjectDet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True):
        super(customREDnetSparseObjectDet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        dimension = 2

        sparse_out_channels = 256

        nPlanes = nr_input_channels
        m = scn.Sequential()

        # Layer 1 - 16 channel convolutions
        out_channels = 16
        m.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, 3, False))
        nPlanes = out_channels
        m.add(scn.BatchNormReLU(nPlanes))

        # Layer 2 - 32 channel convolutions
        out_channels = 32
        m.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, 3, False))
        nPlanes = out_channels
        m.add(scn.BatchNormReLU(nPlanes))

        # Layer 3 - MaxPool
        m.add(scn.MaxPooling(dimension, 3, 2))

        # Layer 4 - 64 channel convolutions
        out_channels = 64
        m.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, 3, False))
        nPlanes = out_channels
        m.add(scn.BatchNormReLU(nPlanes))

        # Layer 5 - 128 channel convolutions
        out_channels = 128
        m.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, 3, False))
        nPlanes = out_channels
        m.add(scn.BatchNormReLU(nPlanes))

        # Layer 6 - MaxPool
        m.add(scn.MaxPooling(dimension, 3, 2))

        # layer 7 - LSTM 256 channels
        out_channels = 256
        m.add(ConvLSTM(dimension, nPlanes, out_channels, 3))

        # layer 8 - output layers
        m.add(scn.SparseToDense(2, sparse_out_channels))

        self.sparseModel = m


        self.cnn_spatial_output_size = [5, 7]
        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = spatial_size_product * 256
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

        return x

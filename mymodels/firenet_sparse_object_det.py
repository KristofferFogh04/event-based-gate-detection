import torch
import torch.nn as nn
import sparseconvnet as scn
from mylayers.conv_GRU import ConvGRU


class FirenetSparseObjectDet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True, freeze_layers=False):
        super(FirenetSparseObjectDet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        dimension = 2
        self.num_recurrent_units = 2

        sparse_out_channels = 256

        nPlanes = nr_input_channels

        # Layer 1 - submanifold convolution
        self.conv1 = scn.Sequential()
        out_channels = 16
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 16
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 2 - submanifold convolution
        self.conv2 = scn.Sequential()
        out_channels = 32
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 32
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 3 - submanifold convolution
        self.conv2 = scn.Sequential()
        out_channels = 64
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 64
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 4 - submanifold convolution
        self.conv2 = scn.Sequential()
        out_channels = 128
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 128
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 5 - submanifold convolution
        self.conv2 = scn.Sequential()
        out_channels = 256
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 256
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels

        # Layer 6 - Gated Recurrent Unit
        kernel_size = 3
        self.GRU1 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)

        # Layer 7 - Gated Recurrent Unit
        kernel_size = 3
        self.GRU2 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)


        # Layer 6 - output layers
        self.sparsetodense = scn.SparseToDense(2, sparse_out_channels)
        self.cnn_spatial_output_size = [5, 7]
        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.spatial_size = self.conv1.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = spatial_size_product * 256
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))
        
        if freeze_layers == True:
            # Load convolutional layers of model
            pth = 'log/pretrained_RNN_backbone_96/checkpoints/model_step_53.pth'
            self.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])
            
            # Freeze convolutional and linear layers
            #for child in self.conv1.children():
            #    for param in child.parameters():
            #        param.requires_grad = False
            #for child in self.linear_1.children():
            #    for param in child.parameters():
            #        param.requires_grad = False
            #for child in self.linear_2.children():
            #    for param in child.parameters():
            #        param.requires_grad = False

    def forward(self, x, prev_states=None):

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        x = self.inputLayer(x)
        x = self.conv1(x)

        x = self.GRU1(x, prev_states[state_idx])
        state_idx += 1
        states.append(x)

        x = self.GRU2(x, prev_states[state_idx])
        state_idx += 1
        states.append(x)

        x = self.sparsetodense(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

        return x, states
    
    """ Original firenet architecture. Probably not suitable for object detection
class FirenetSparseObjectDet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True):
        super(FirenetSparseObjectDet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        dimension = 2
        self.num_recurrent_units = 2

        sparse_out_channels = 256

        nPlanes = nr_input_channels

        # Layer 1 - submanifold convolution
        self.conv1 = scn.Sequential()
        out_channels = 16
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))


        # Layer 2 - Gated Recurrent Unit
        out_channels = 16
        kernel_size = 3
        self.GRU1 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)
        nPlanes = out_channels

        # Layer 3 - Submanifold convolution
        self.conv2 = scn.Sequential()
        out_channels = 16
        kernel_size = 3
        self.conv2.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv2.add(scn.BatchNormReLU(nPlanes))
        self.conv2.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))

        # Layer 4 - Gated Recurrent Unit
        out_channels = 16
        kernel_size = 3
        self.GRU2 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)
        nPlanes = out_channels

        # Layer 5 - Submanifold convolution
        self.conv3 = scn.Sequential()
        out_channels = 16
        kernel_size = 3
        self.conv3.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv3.add(scn.BatchNormReLU(nPlanes))
        self.conv3.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))

        # Layer 6 - output layers
        self.sparsetodense = scn.SparseToDense(2, sparse_out_channels)
        self.cnn_spatial_output_size = [5, 7]
        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.spatial_size = self.conv1.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = spatial_size_product * 256
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))

    def forward(self, x, prev_states):

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        x = self.inputLayer(x)
        x = self.conv1(x)

        x = self.GRU1(x, prev_states[state_idx])
        state_idx += 1
        states.append(x)

        x = self.conv2(x)

        x = self.GRU2(x, prev_states[state_idx])
        state_idx += 1
        states.append(x)

        x = self.conv3(x)
        x = self.sparsetodense(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

        return x, states
    """

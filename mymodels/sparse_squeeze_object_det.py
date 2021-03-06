import torch
import torch.nn as nn
import sparseconvnet as scn
from mylayers.conv_GRU import ConvGRU


class SparseSqueezeNet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True, freeze_layers=False):
        super(SparseSqueezeNet, self).__init__()
        self.pretrain = True
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        dimension = 2
        self.num_recurrent_units = 2

        sparse_out_channels = 256

        nPlanes = nr_input_channels

        # Layer 1 - submanifold convolution
        self.conv1 = scn.Sequential()
        out_channels = 64
        kernel_size = 3
        self.conv1.add(scn.Convolution(2, nPlanes, out_channels, kernel_size, filter_stride=2, bias=False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels

        # Layer 2 - Fire layers
        out_channels1 = 16
        out_channels2 = 64
        out_channels3 = 64
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels

        # Layer 3 - Fire layers
        out_channels1 = 32
        out_channels2 = 128
        out_channels3 = 128
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels

        # Layer 4 - Fire layers
        out_channels1 = 64
        out_channels2 = 256
        out_channels3 = 256
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        nPlanes = self.fire_layer( dimension, nPlanes, out_channels1, out_channels2, out_channels3)
        nPlanes = out_channels
        
        # Layer 6 - strided convolution
        out_channels = 256
        kernel_size = 3
        self.conv1.add(scn.Convolution(2, nPlanes, out_channels, kernel_size, filter_stride=2, bias=False))
        self.conv1.add(scn.BatchNormReLU(out_channels))

        # Layer 6 - Gated Recurrent Unit
        kernel_size = 3
        self.GRU1 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)


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
            pth = 'log/RNN_TBPTT_trained_run5_manyepochs_93/checkpoints/model_step_39.pth'
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

        if self.pretrain:
            x = self.inputLayer(x)
            x = self.conv1(x)

            x = self.sparsetodense(x)
            x = x.view(-1, self.linear_input_features)
            x = self.linear_1(x)
            x = torch.relu(x)
            x = self.linear_2(x)
            x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

            return x

        else:
            if prev_states is None:
                prev_states = [None] * (self.num_recurrent_units)

            states = []
            state_idx = 0

            x = self.inputLayer(x)
            x = self.conv1(x)

            x = self.GRU1(x, prev_states[state_idx])
            state_idx += 1
            states.append(x)

            x = self.sparsetodense(x)
            x = x.view(-1, self.linear_input_features)
            x = self.linear_1(x)
            x = torch.relu(x)
            x = self.linear_2(x)
            x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

            return x, states


    def fire_layer(self, dimension, nPlanes, out_channels1, out_channels2, out_channels3):

        squeeze1x1 = self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels1, 1, False))

        expand1x1 = self.conv1.add(scn.SubmanifoldConvolution(dimension, out_channels1, out_channels2, 1, False))

        expand3x3 = self.conv1.add(scn.SubmanifoldConvolution(dimension, out_channels2, out_channels3, 3, False))

        return out_channels3

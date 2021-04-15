import torch
import torch.nn as nn
import torch.nn.functional as f 
from torch.nn import init

import sparseconvnet as scn


class ConvGRU(nn.Module):
    """
    Generate a sparse convolutional GRU cell. 
    Adapted from https://github.com/cedric-scheerlinck/rpg_e2vid/blob/cedric/firenet/model/submodules.py
    """

    def __init__(self, dimension, input_size, hidden_size, kernel_size):
            super().__init__()
            padding = kernel_size // 2
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.reset_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, False)
            self.update_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, False)
            self.out_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, False)

            init.orthogonal_(self.reset_gate.weight)
            init.orthogonal_(self.update_gate.weight)
            init.orthogonal_(self.out_gate.weight)
            init.constant_(self.reset_gate.bias, 0.)
            init.constant_(self.update_gate.bias, 0.)
            init.constant_(self.out_gate.bias, 0.)
            
        def input_spatial_size(self, out_size):
            return out_size

        def forward(self, input_, prev_state):

            # get batch and spatial sizes
            batch_size = input_.data.size()[0]
            spatial_size = input_.data.size()[2:]

            # generate empty prev_state, if None is provided
            if prev_state is None:
                state_size = [batch_size, self.hidden_size] + list(spatial_size)
                prev_state = torch.zeros(state_size).to(input_.device)

            # data size is [batch, channel, height, width]
            stacked_inputs = torch.cat([input_, prev_state], dim=1)
            update = torch.sigmoid(self.update_gate(stacked_inputs))
            reset = torch.sigmoid(self.reset_gate(stacked_inputs))
            out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
            new_state = prev_state * (1 - update) + out_inputs * update

            return new_state



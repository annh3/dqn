import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear


from configs.q4_nature import config

from collections import OrderedDict

import pdb


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """ 
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        print("img_height: ", img_height)
        print("img_width: ", img_width)
        print("n_channels: ", n_channels)
        print("n_actions: ", num_actions)

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
        modules = OrderedDict()

        # def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1): -> h,w 
        pad1 = (3 * img_height - 4 + 8) // 2
        modules['layer1'] = nn.Conv2d(n_channels * self.config.state_history, 32, 8, stride=4, padding=pad1)
        modules['relu1'] = nn.ReLU()
        out1h, out1w = conv_output_shape((img_height, img_width), kernel_size=8, stride=4, pad=pad1, dilation=1)

        pad2 = (out1w + 2) // 2
        modules['layer2'] = nn.Conv2d(32, 64, 4, stride=2, padding=pad2)
        modules['relu2'] = nn.ReLU()
        out2h, out2w = conv_output_shape((out1h, out1w), kernel_size=4, stride=2, pad=pad2, dilation=1)

        pad3 = 1
        modules['layer3'] = nn.Conv2d(64, 64, 3, stride=1, padding=pad3)
        modules['relu3'] = nn.ReLU()
        out3h, out3w = conv_output_shape((out2h, out2w), kernel_size=3, stride=1, pad=pad3, dilation=1)

        modules['flatten'] = nn.Flatten()
        modules['fc'] = nn.Linear(out3h*out3w*64, num_actions)
        self.q_network = nn.Sequential(modules)

        target_modules = OrderedDict()
        target_modules['t_layer1'] = nn.Conv2d(n_channels * self.config.state_history, 32, 8, stride=4, padding=pad1)
        target_modules['t_relu1'] = nn.ReLU()
        target_modules['t_layer2'] = nn.Conv2d(32, 64, 4, stride=2, padding=pad2)
        target_modules['t_relu2'] = nn.ReLU()
        target_modules['t_layer3'] = nn.Conv2d(64, 64, 3, stride=1, padding=pad3)
        target_modules['t_relu3'] = nn.ReLU()
        target_modules['t_flatten'] = nn.Flatten()
        target_modules['t_fc'] = nn.Linear(out3h*out3w*64, num_actions)
        self.target_network = nn.Sequential(target_modules)

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        state = state.permute(0, 3, 1, 2)
        #pdb.set_trace()
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        if network == 'q_network':
            out = self.q_network(state)
        else:
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)

"""
This is companion code to Project 4 for CSEP576au21
(https://courses.cs.washington.edu/courses/csep576/21au/)

Instructor: Vitaly Ablavsky
"""

# ======================================================================
# Copyright 2021 Vitaly Ablavsky https://corvidim.net/ablavsky/
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ======================================================================


from pdb import set_trace as keyboard

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))



#########################################################################
#                        PoseRegression_v1()
#########################################################################
class PoseRegression_v1(nn.Module):
    """
    This class defines a rather toy (shallow) network architecture, but
    it turns out to be sufficient for our (toy) pose-regression task.
    ....
    
    Note to students/TAs
    * add dropout
    * implement L2 normalization layer (if predicting [cos(t), sin(t)])

    """
    '''def __init__(self, par):
        super(PoseRegression_v1, self).__init__()
        # the next two attributes are required by train_util.perform_testing()
        self.conv_feats = True
        self.regression_problem=True
        self.mode = par['mode']

        self.out_channels = 64
        self.n_hidden = 128
        self.n_out = par['n_out']
        self.m_regress = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32,32, 3,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, self.out_channels, 3, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.LazyLinear(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(self.n_out),)'''
    
    ### Updated model to modify first convolution layer kernel size and add batch normalization to it.
    def __init__(self, par):
        super(PoseRegression_v1, self).__init__()
        # the next two attributes are required by train_util.perform_testing()
        self.conv_feats = True
        self.regression_problem=True
        self.mode = par['mode']

        self.out_channels = 64
        self.n_hidden = 128
        self.n_out = par['n_out']
        self.m_regress = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32,32, 3,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, self.out_channels, 3, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.LazyLinear(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(self.n_out),)

    def forward(self, x):
        output = self.m_regress(x)
        if self.mode == 'train_and_test_2_out':
            output = F.normalize(output)
        return output
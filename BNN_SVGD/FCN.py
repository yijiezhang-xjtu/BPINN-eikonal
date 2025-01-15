# scitific cal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
from time import time
import sys
import os
import gc
import subprocess # Call the command line
from subprocess import call
import pdb
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import the modules you need
#import foamFileOperation as foamOp
class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
            
class Net(torch.nn.Module):
    def __init__(self, n_feature1, n_hidden1,  n_feature2, n_hidden2):
        super(Net, self).__init__()
        self.features1 = nn.Sequential()
        self.features1.add_module('hidden0', torch.nn.Linear(n_feature1, n_hidden1))
        self.features1.add_module('active0', Swish())
        self.features1.add_module('hidden1', torch.nn.Linear(n_hidden1, n_hidden1))
        self.features1.add_module('active1', Swish())
        self.features1.add_module('hidden2', torch.nn.Linear(n_hidden1, n_hidden1))
        self.features1.add_module('active2', Swish())
        self.features1.add_module('hidden3', torch.nn.Linear(n_hidden1, n_hidden1))
        self.features1.add_module('active3', Swish())
        self.features1.add_module('predict', torch.nn.Linear(n_hidden1, 1))
        
        self.features2 = nn.Sequential()
        self.features2.add_module('hidden0', torch.nn.Linear(n_feature2, n_hidden2))
        self.features2.add_module('active0', nn.ELU())
        self.features2.add_module('hidden1', torch.nn.Linear(n_hidden2, n_hidden2))
        self.features2.add_module('active1', nn.ELU())
        self.features2.add_module('hidden2', torch.nn.Linear(n_hidden2, n_hidden2))
        self.features2.add_module('active2', nn.ELU())
        self.features2.add_module('hidden3', torch.nn.Linear(n_hidden2, n_hidden2))
        self.features2.add_module('active3', nn.ELU())
        self.features2.add_module('predict', torch.nn.Linear(n_hidden2, 1))
        # self.features2 = nn.Sequential()
        # self.features2.add_module('hidden0', torch.nn.Linear(n_feature2, n_hidden2))
        # self.features2.add_module('active0', nn.SELU())
        # self.features2.add_module('hidden1', torch.nn.Linear(n_hidden2, n_hidden2))
        # self.features2.add_module('active1', nn.SELU())
        # self.features2.add_module('hidden2', torch.nn.Linear(n_hidden2, n_hidden2))
        # self.features2.add_module('active2', nn.SELU())
        # self.features2.add_module('hidden3', torch.nn.Linear(n_hidden2, n_hidden2))
        # self.features2.add_module('active3', nn.SELU())
        # self.features2.add_module('predict', torch.nn.Linear(n_hidden2, 1))
    def forward(self, xsx, x):
        y1 = self.features1(xsx)
        y2 = self.features2(x)
        output2 = 1 / (1 + torch.exp(-y2))
        output = torch.cat((y1,output2),1)
        return output
    
    def reset_parameters(self, verbose=False):
        #TODO: where did you define module?
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
        if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):


                module.reset_parameters()
            if verbose:
                print("Reset parameters in {}".format(module))


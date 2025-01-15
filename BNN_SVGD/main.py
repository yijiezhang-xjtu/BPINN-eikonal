import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
#from time import time
import time
import sys
import os
import gc
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
# local import
import FCN
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from SVGD import SVGD
from args import args, device
from cases import cases

# Specific hyperparameter for SVGD
n_samples = args.n_samples
noise = args.noise

# deterministic NN structure
nFeature1 = 4
nNeuron1 = 20
nFeature2 = 2
nNeuron2 = 10

denseFCNN = FCN.Net(nFeature1,nNeuron1,nFeature2,nNeuron2)
#print(denseFCNN)

# Bayesian NN
bayes_nn = BayesNN(denseFCNN, n_samples=n_samples, noise=noise).to(device)
# specifying training case
train_case  = cases(bayes_nn)
#time.sleep(10)
train_loader, train_size = train_case.dataloader()
ntrain = train_size

# Initialize SVGD
svgd = SVGD(bayes_nn,train_loader)
print('Start training.........................................................')
tic = time.time()
epochs = args.epochs
LOSSr,LOSSv,LOSStau = [], [], []

"""
To start training, uncomment the code below
"""
# for epoch in range(epochs):
#     LOSSr,LOSSv,LOSStau = svgd.train(epoch, LOSSr,LOSSv,LOSStau)
# training_time = time.time() - tic
# print('finished in ',training_time)
# torch.save(bayes_nn.state_dict(),"test1e-2_1p.pt")
# #np.savetxt('datalikelihood.csv',data_likelihod)
# #np.savetxt('eqlikelihood.csv',eq_likelihood)
# LOSSr = np.array(torch.tensor(LOSSr, device='cpu'))
# LOSSv = np.array(torch.tensor(LOSSv, device='cpu'))
# LOSStau = np.array(torch.tensor(LOSStau, device='cpu'))
# np.savetxt('LOSSr.csv',LOSSr)
# np.savetxt('LOSSv.csv',LOSSv)
# np.savetxt('LOSStau.csv',LOSStau)

# plot
train_case.plot()


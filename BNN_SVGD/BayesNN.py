import numpy
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
import pdb
import subprocess # Call the command line
from subprocess import call
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
# local import
from FCN import Net
from args import args, device


class BayesNN(nn.Module):
    """Define Bayesian netowrk

    """
    def __init__(self, model, n_samples=2, noise=1e-6):
        super(BayesNN, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(
                torch.typename(model)))
        self.n_samples = n_samples # number of particles (# of perturbed NN)
        print('n_samples is',n_samples)
        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # for efficiency, represent StudentT params using Gamma params
        self.w_prior_shape = 1.
        self.w_prior_rate = 0.05 

        # noise variance 1e-6: beta ~ Gamma(beta | shape, rate)
        self.beta_prior_shape = (2.)
        self.beta_prior_rate = noise

        ################
        # for the equation loglilihood
        self.var_eq = 1e-4

        ################
        # replicate `n_samples` instances with the same network as `model`
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(model)
            #new_instance = Net(1, 20)
            # initialize each model instance with their defualt initialization
            # instead of the prior
            #new_instance.reset_parameters()
            def init_normal(m):
                if type(m) == nn.Linear:
                    nn.init.kaiming_normal_(m.weight)
            new_instance.apply(init_normal)
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
            #t.sleep(100)
        
        self.nnets = nn.ModuleList(instances)
        #del instances # delete instances


    
    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            #print(name,param)
            count += param.numel()#返回数组中元素的个数
        return count

    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta1.item() for i in range(self.n_samples)], 
                            device=device),torch.tensor([self.nnets[i].log_beta2.item() for i in range(self.n_samples)], 
                            device=device)

    
    def forward(self, inputs_xsx, inputs_x):

        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(inputs_xsx, inputs_x))
        output = torch.stack(output)

        return output
    def _log_joint(self, index, output, target, ntrain):
        """Log joint probability or unnormalized posterior for single model
        instance. Ignoring constant terms for efficiency.
        Can be implemented in batch computation, but memory is the bottleneck.
        Thus here we trade computation for memory, e.g. using for loop.
        Args:
            index (int): model index, 0, 1, ..., `n_samples`
            output (Tensor): y_pred
            target (Tensor): y
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob
        Returns:
            Log joint probability (zero-dim tensor)
        """
        # Normal(target | output, 1 / beta * I)


        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * 1e4 * (target[:,0] - output[:,0]).pow(2).sum())

        log_likelihood += 0.1 * ntrain / output.size(0) * (- 0.5 * 1e3 * (target[:,1] - output[:,1]).pow(2).sum())


        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.).to(device)
        for param in self.nnets[index].features1.parameters():
            log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        for param in self.nnets[index].features2.parameters():
            log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        # log prob of prior of log noise-precision (NOT noise precision)

        return log_likelihood + log_prob_prior_w


    def _mse(self, index, output, target, ntrain):

        loss_f = nn.MSELoss()
        loss = loss_f(output,target)

        return loss
    def criterion(self,index,xs,zs,x,z,vs,ntrain,vmin,vmax):
        xs = torch.FloatTensor(xs).to(device)
        zs = torch.FloatTensor(zs).to(device)
        x1 = torch.FloatTensor(x).to(device)
        z1 = torch.FloatTensor(z).to(device)
        x2 = torch.FloatTensor(x).to(device)
        z2 = torch.FloatTensor(z).to(device)
        vs = torch.FloatTensor(vs).to(device)

        xs.requires_grad = True
        zs.requires_grad = True
        x1.requires_grad = True
        z1.requires_grad = True
        x2.requires_grad = True
        z2.requires_grad = True
        
        T0 = torch.sqrt((z1-zs)**2+(x1-xs)**2).div(vs)


        px0 = (x1-xs).div(T0*vs**2)
        pz0 = (z1-zs).div(T0*vs**2) 

        net_in_xsx = torch.cat((xs,zs,x1,z1),1)
        net_in_x = torch.cat((x2,z2),1)

        output = self.nnets[index].forward(net_in_xsx,net_in_x)

        tau = output[:,0]
        v = output[:,1]*(vmax-vmin)+vmin

        tau = tau.view(len(tau),-1)
        v = v.view(len(v),-1)
        T0 = T0.view(len(T0),-1)

        a = 1e5

        tau_x = torch.autograd.grad(tau,x1,grad_outputs=torch.ones_like(x1),create_graph = True,only_inputs=True)[0]
        tau_z = torch.autograd.grad(tau,z1,grad_outputs=torch.ones_like(x1),create_graph = True,only_inputs=True)[0]
        loss = (T0*tau_x+tau*px0)**2 + (T0*tau_z+tau*pz0)**2 - 1/(v)**2

        logloss = ntrain / output.size(0) * (
                            - 0.5 * a* (loss - torch.zeros_like(loss)).pow(2).sum()
                            )

        return logloss,loss

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
import copy
import math
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
from time import time
import time as t
import sys
import os
import gc

# import the modules you need
#import foamFileOperation as foamOp
from FCN import Net
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from args import args, device
import skfmm

n_samples = args.n_samples
lr = args.lr
lr_noise = args.lr_noise
ntrain = args.ntrain
class SVGD(object):
    """
    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn,train_loader):
        """
        For-loop implementation of SVGD.
        Args:
            bayes_nn (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)
        """
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.optimizers = self._optimizers_schedulers(lr, lr_noise)
    def _squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2
        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of 
                one sample
        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)
    


    def _Kxx_dxKxx(self, X):
        
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.
        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        # print("squared_dist",squared_dist)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx


        '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta) #矩阵X样本之间（m*n）的欧氏距离(2-norm) ，返回值为 Y (m*m)为压缩距离元组或矩阵
        pairwise_dists = squareform(sq_dist)**2 #用来把一个向量格式的距离向量转换成一个方阵格式的距离矩阵，反之亦然
        # print('shape of theta is',self.theta.shape)
        # print('shape of sq_dist is',sq_dist.shape)
        # print('shape of pairwise_dists is',pairwise_dists.shape)
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)
        #print('Kxy.shape is',Kxy.shape)
        time.sleep(1)
        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        #print('dxkxy.shape is',dxkxy.shape)
        time.sleep(1)
        return (Kxy, dxkxy)
    



    def _optimizers_schedulers(self, lr, lr_noise):
        """Initialize Adam optimizers and schedulers (ReduceLROnPlateau)
        Args:
            lr (float): learning rate for NN parameters `w`
            lr_noise (float): learning rate for noise precision `log_beta`
        """
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            parameters = [{'params': self.bayes_nn[i].features1.parameters()}, 
                          {'params': self.bayes_nn[i].features2.parameters()}]

            optimizer_i = torch.optim.Adam(parameters, lr=lr)
            optimizers.append(optimizer_i)

        return optimizers
    

    def train(self, epoch, LOSSr, LOSSv, LOSStau):
        self.bayes_nn.train()
        mse = 0.       

        data_xrx = np.load('data_XrX.npz')
        data_xsx = np.load('data_XsX.npz')
        data_target = np.load('targetref.npz')
        normalize = np.load('normalize.npz')
        
        Xs = data_xsx['Xs']
        Zs = data_xsx['Zs']
        Vs = torch.Tensor(data_xsx['Vs']).to(device)
        tauref = data_target['tauref']
        Vref = data_target['Vref']
        T0 = data_target['T0']
        
        Xsi = torch.Tensor(data_xrx['Xs_a']).to(device)
        Zsi = torch.Tensor(data_xrx['Zs_a']).to(device)
        Xrr = torch.Tensor(data_xrx['Xr_a']).to(device)
        Zrr = torch.Tensor(data_xrx['Zr_a']).to(device)
        #Zs = torch.Tensor(Zs).to(device)
        #noise_lv = 0.0
        #Vref, Tref = self._addnoise(noise_lv, Vref, Tref)
        velmodel_min = float(normalize['velmodel_min'])
        velmodel_max = float(normalize['velmodel_max'])


        T0 = torch.Tensor(T0).to(device)

        for batch_idx, (xs, zs, x, z, vs) in enumerate(self.train_loader):
            Xrr1,Xrr2 = Xrr.reshape(-1,1),Xrr.reshape(-1,1)
            Zrr1,Zrr2 = Zrr.reshape(-1,1),Zrr.reshape(-1,1)
            Xrr1.requires_grad = True
            Zrr1.requires_grad = True
            Xsi.requires_grad = True
            Zsi.requires_grad = True
            Xrr2.requires_grad = True
            Zrr2.requires_grad = True
            sp_inputs_xsx = torch.cat((Xsi,Zsi,Xrr1,Zrr1),1)
            sp_inputs_x = torch.cat((Xrr2,Zrr2),1)

            noise_lv = 0.05
            tau_n, V_n = self._addnoise(noise_lv,tauref, Vref)
            V_n = torch.Tensor(V_n).to(device)
            tau_n = torch.Tensor(tau_n).to(device)
            # V_n = torch.Tensor(Vref).to(device)
            # tau_n = torch.Tensor(tauref).to(device)
            
            sp_target = torch.cat((tau_n,V_n),1)
            ## paste outlet
            self.bayes_nn.zero_grad()
            
            
            output = torch.zeros_like(x).to(device)
            sp_output = torch.zeros_like(sp_target).to(device)

            # all gradients of log joint probability: (S, P)
            grad_log_joint = []
            # all model parameters (particles): (S, P)
            theta = []
            # store the joint probabilities
            log_joint = 0.
            lossr,lossv,losstau = 0, 0, 0
            for i in range(self.n_samples):
                #####################
                ###modified for sparse data stenosis
                ## forward for training data
                sp_output_i = self.bayes_nn[i].forward(sp_inputs_xsx, sp_inputs_x)

                ## loss for unlabelled points
                log_eq_i,loss_i =  self.bayes_nn.criterion(i,xs,zs,x,z,vs,ntrain,velmodel_min,velmodel_max)
                ## loss for labelled points
                log_joint_i = self.bayes_nn._log_joint(i, sp_output_i, sp_target, ntrain)

                ### for monity purpose
                lossr, lossv, losstau = self._monitor(lossr, loss_i, lossv, losstau, sp_output_i, sp_target)
                ### 
                log_joint_i += log_eq_i
                log_joint_i.backward()

                #####
                # backward frees memory for computation graph\

                # computation below does not build computation graph
                # extract parameters and their gradients out from models
                vec_param, vec_grad_log_joint = parameters_to_vector(
                    self.bayes_nn[i].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                # print(vec_param.shape)
                # print(vec_param.unsqueeze(0).shape)
                theta.append(vec_param.unsqueeze(0))

            # calculating the kernel matrix and its gradients
            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            grad_logp = torch.mm(Kxx, grad_log_joint)
            # negate grads here!!!
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            ## switch back to 1 particle
            #grad_theta = grad_log_joint
            # explicitly deleting variables does not release memory :(
       
            # update param gradients
            for i in range(self.n_samples):
                vector_to_parameters(grad_theta[i,:],
                    self.bayes_nn[i].parameters(), grad=True)
                
                self.optimizers[i].step()
            # WEAK: no loss function to suggest when to stop or
            # approximation performance
            #mse = F.mse_loss(output / self.n_samples, target).item()

            lossr/= self.n_samples
            lossv/=self.n_samples
            losstau/=self.n_samples
            #print('len(self.train_loader)',len(self.train_loader))
            if batch_idx % 50 ==0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvgLossr: {:.10f}\tAvgLossv: {:.10f}\tAvgLosstau: {:.10f}'.format(
                    epoch, batch_idx * len(x), len(self.train_loader)*len(x),
                    100. * batch_idx / len(self.train_loader), lossr.item(),lossv.item(),losstau.item()))
                LOSSr.append(lossr)
                LOSSv.append(lossv)
                LOSStau.append(losstau)


        if epoch%100==0:
            self._savept(epoch)
        return LOSSr,LOSSv,LOSStau

    def _addnoise(self, noise_lv, tauref, Vref):
        Vref = torch.Tensor(Vref).cpu()
        tauref = torch.Tensor(tauref).cpu()
        V_noise = torch.normal(0, noise_lv*Vref)
        tau_noise = torch.normal(0, noise_lv*tauref)
        Vref += V_noise
        tauref += tau_noise

        return tauref.reshape(-1,1), Vref.reshape(-1,1)

    
    def _monitor(self, lossr, loss_i, lossv, losstau, sp_output_i, sp_target):
        loss_f = nn.MSELoss()
        lossr += loss_f(loss_i,torch.zeros_like(loss_i))
        lossv += loss_f(sp_output_i[:,1], sp_target[:,1])
        losstau += loss_f(sp_output_i[:,0], sp_target[:,0])
        
        return lossr, lossv, losstau

    def _savept(self, epoch):
        torch.save(self.bayes_nn.state_dict(),"test"+str(epoch)+"0.05_test1.pt")



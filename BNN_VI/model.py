import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from args import args, device

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
            
            
class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))        

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to(device)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(device)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)

class MLP_BBB(nn.Module):
    def __init__(self, noise_tol=.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        
        # only input and output layer are be considered as Bayesian model
        self.hiddent = nn.Sequential()
        self.hiddent.add_module('hidden', Linear_BBB(4, 20, prior_var=prior_var))
        self.hiddent.add_module('active', Swish())
        self.featuret = nn.Sequential()
        self.featuret.add_module('hidden1', torch.nn.Linear(20, 20))
        self.featuret.add_module('active1', Swish())
        self.featuret.add_module('hidden2', torch.nn.Linear(20, 20))
        self.featuret.add_module('active2', Swish())
        self.featuret.add_module('hidden3', torch.nn.Linear(20, 20))
        self.featuret.add_module('active3', Swish())
        self.predictt = Linear_BBB(20, 1, prior_var=prior_var)

        self.hiddenv = nn.Sequential()
        self.hiddenv.add_module('hidden', Linear_BBB(2, 10, prior_var=prior_var))
        self.hiddenv.add_module('active', nn.SELU())
        self.featurev = nn.Sequential()
        self.featurev.add_module('hidden1', torch.nn.Linear(10, 10))
        self.featurev.add_module('active1', nn.SELU())
        self.featurev.add_module('hidden2', torch.nn.Linear(10, 10))
        self.featurev.add_module('active2', nn.SELU())
        self.featurev.add_module('hidden3', torch.nn.Linear(10, 10))
        self.featurev.add_module('active3', nn.SELU())
        self.predictv = Linear_BBB(10, 1, prior_var=prior_var)

        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

    def forward(self, xsx, x):
        # again, this is equivalent to a standard multilayer perceptron
        tau = self.hiddent(xsx)
        tau = self.featuret(tau)
        tau = self.predictt(tau)

        v = self.hiddenv(x)
        v = self.featurev(v)
        v = self.predictv(v)
        v = torch.sigmoid(v)
        output = torch.cat((tau,v),1)
        return output

    def log_prior(self):
        # calculate the log prior over all the layers

        hiddent_log_prior = self.hiddent.hidden.log_prior
        hiddent_log_prior += self.predictt.log_prior
        
        hiddenv_log_prior = self.hiddenv.hidden.log_prior
        hiddenv_log_prior += self.predictv.log_prior

        return hiddent_log_prior + hiddenv_log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        
        hiddent_log_post = self.hiddent.hidden.log_post
        hiddent_log_post += self.predictt.log_post
        
        hiddenv_log_post = self.hiddenv.hidden.log_post
        hiddenv_log_post += self.predictv.log_post
        
        return hiddent_log_post + hiddenv_log_post

    def sample_elbo(self, net_in_xsx, net_in_x, target, samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors

        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples).cuda()
        # logb_likes = torch.zeros(samples).cuda()
        
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            output = self.forward(net_in_xsx, net_in_x)
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = ((101*101)/output.size(0))*Normal(output[:,0], 0.01).log_prob(target[:,0]).sum() # calculate the log likelihood
            log_likes[i] += ((101*101)/output.size(0))*Normal(output[:,1], 0.1).log_prob(target[:,1]).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        # calculate the negative elbo (which is used in our loss function)
        return log_post - log_prior - log_like
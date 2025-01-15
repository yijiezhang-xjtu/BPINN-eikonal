import argparse
import torch
import json
import random
from pprint import pprint
#from utils.misc import mkdirs
import time

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='FCN with SVGD')       

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=5, help='(5-30) number of model instances in SVGD')        

        # data
        self.add_argument('--ntrain', type=int, default=101*101, help="number of training data")

        
        # training
        self.add_argument('--epochs', type=int, default=1001, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-3, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=1e-4, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=1024, help='batch size for training')
        self.add_argument('--seed', type=int, default=123, help='manual seed used in Tensor')
        self.add_argument('--noise', type=float, default=1e-6, help='noise in prior')


    def parse(self):
        args = self.parse_args()

        print('Arguments:')
        pprint(vars(args))
        

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from FCN import Net
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from SVGD import SVGD
from args import args,device
import random
import skfmm

random.seed(20)
class cases:
	def __init__(self,bayes_nn):
		self.epochs = args.epochs
		self.bayes_nn = bayes_nn
	
	def dataloader(self):
		train_size = args.batch_size
		v0 = 2.; # Velocity at the origin of the model


		zmin = 0.; zmax = 2.; deltaz = 0.02;
		xmin = 0.; xmax = 2.; deltax = 0.02;


		# Point-source location
		sz = 1.0; sx = 1.0;

		z = np.arange(zmin,zmax+deltaz,deltaz)
		nz = z.size

		x = np.arange(xmin,xmax+deltax,deltax)
		nx = x.size


		Z,X = np.meshgrid(z,x,indexing='ij')
		velmodel = v0*np.ones_like(Z)

		for i in range(len(velmodel)):
		    for j in range(len(velmodel[0])):
		        if ((i*0.02-1)/0.4)**2 + ((j*0.02-1)/0.6)**2 <= 1:
		            velmodel[i][j] = 3

		selected_pts1_1 = np.linspace(0, Z.size-101, num = 5).astype('int64')
		selected_pts1_2 = np.linspace(100, Z.size-1, num = 5).astype('int64')

		selected_pts1 = np.append(selected_pts1_1, selected_pts1_2)

		selected_pts2 = np.linspace(0, Z.size-1, Z.size).astype('int64')
		selected_pts2 = np.setdiff1d(selected_pts2,selected_pts1)



		Zsin = Z.reshape(-1)[selected_pts1]
		Xsin = X.reshape(-1)[selected_pts1]

		Zi = Z.reshape(-1)[selected_pts2]
		Xi = X.reshape(-1)[selected_pts2]

		Z,X,Zs,Xs,vs = [],[],[],[],[]

		for ns, (szi, sxi) in enumerate(zip(Zsin, Xsin)):
			Zsi = szi*np.ones_like(Zi)
			Xsi = sxi*np.ones_like(Xi)

			v_si = velmodel[int(szi/deltaz),int(sxi/deltax)]
			vsi = v_si*np.ones_like(Xi)
			Z = np.concatenate((Z,Zi),0)
			X = np.concatenate((X,Xi),0)
			Zs = np.concatenate((Zs,Zsi),0)
			Xs = np.concatenate((Xs,Xsi),0)
			vs = np.concatenate((vs,vsi),0)

		data = torch.utils.data.TensorDataset(torch.FloatTensor(Xs.reshape(-1,1)), torch.FloatTensor(Zs.reshape(-1,1)),torch.FloatTensor(X.reshape(-1,1)), torch.FloatTensor(Z.reshape(-1,1)), torch.FloatTensor(vs.reshape(-1,1)))
		train_loader = torch.utils.data.DataLoader(data, batch_size=train_size, shuffle=False)

		print('len(data is)', len(data), x.shape)
		print('len(dataloader is)', len(train_loader))
		return train_loader,train_size

	def plot(self):
		# plot the result
		self.bayes_nn.load_state_dict(torch.load("test7000.05_test1.pt",map_location = 'cpu'))
		self.bayes_nn.eval()


		v0 = 2.; # Velocity at the origin of the model

		zmin = 0.; zmax = 2.; deltaz = 0.02;
		xmin = 0.; xmax = 2.; deltax = 0.02;


		# Point-source location
		sz = 1.0; sx = 1.0;


		z = np.arange(zmin,zmax+deltaz,deltaz)
		nz = z.size

		x = np.arange(xmin,xmax+deltax,deltax)
		nx = x.size


		Z,X = np.meshgrid(z,x,indexing='ij')
		# Preparing velocity model

		velmodel = v0*np.ones_like(Z)

		for i in range(len(velmodel)):
		    for j in range(len(velmodel[0])):
		        if ((i*0.02-1)/0.4)**2 + ((j*0.02-1)/0.6)**2 <= 1:
		            velmodel[i][j] = 3


		normalize = np.load('normalize.npz')
		velmodel_min = float(normalize['velmodel_min'])
		velmodel_max = float(normalize['velmodel_max'])

		Z,X = np.meshgrid(z,x,indexing='ij')
		sz = 1.
		sx = 0.

		xt,zt = torch.Tensor(X), torch.Tensor(Z)
		xt,zt = xt.view(-1,1),zt.view(-1,1)
		xt.requires_grad = True
		zt.requires_grad = True
		xs = sx*torch.ones_like(xt)
		zs = sz*torch.ones_like(xt)
		xt, zt = xt.to(device), zt.to(device)
		xs, zs = xs.to(device), zs.to(device)
		inputs_xsx = torch.cat((xs,zs,xt,zt),1)
		inputs_x = torch.cat((xt,zt),1)

		print('inputs is',inputs_xsx.shape,inputs_x.shape)
		y_pred_mean = self.bayes_nn.forward(inputs_xsx,inputs_x)
		#pred = y_pred_mean.cpu().detach().numpy()
		pred = y_pred_mean
		pred[:,:,1] = pred[:,:,1]*(velmodel_max-velmodel_min)+velmodel_min

		vs = velmodel[int(sz/deltaz),int(sx/deltax)]
		phi = -1*np.ones_like(X)
		phi[np.logical_and(Z==sz, X==sx)] = 1
		d = skfmm.distance(phi, dx=2e-2)
		T_data = skfmm.travel_time(phi, velmodel, dx=2e-2)

		print('pred.shape is',pred.shape)
		mean = pred.mean(0)
		EyyT = (pred ** 2).mean(0)
		EyEyT = mean ** 2
		print(EyyT.shape)

		var_tau = EyyT[:,0] - EyEyT[:,0]
		var_v = 0.00001 + (EyyT[:,1] - EyEyT[:,1])
		print('mean.shape',mean.shape)

		tau_hard = mean[:,0]
		v_hard = mean[:,1]
		tau_hard = tau_hard.view(len(tau_hard),-1)

		v_hard = v_hard.view(len(v_hard),-1)

		tau_hard = tau_hard.cpu().detach().numpy()
		v_hard = v_hard.cpu().detach().numpy()

		tau_pred = tau_hard.reshape(101,101)
		T0 = np.sqrt((Z-sz)**2+(X-sx)**2)/vs
		T_pred = tau_pred*T0
		tautrue = np.divide(T_data, T0, out=np.ones_like(T0), where=T0!=0)
		taudiff = np.abs(tautrue-tau_pred)

		var_tau = var_tau.view(len(var_tau),-1)
		print('var_tau',max(var_tau),var_tau.mean())
		var_v = var_v.view(len(var_v),-1)
		print('var_v',max(var_v),var_v.mean())
		var_tau = var_tau.cpu().detach().numpy()
		var_v = var_v.cpu().detach().numpy()

		plt.style.use('default')
		plt.figure(figsize=(5,5))
		ax = plt.gca()
		im = ax.imshow(np.abs(v_hard.reshape(velmodel.shape)), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
		plt.xlabel('X (km)', fontsize=11)
		plt.xticks(fontsize=10)
		plt.ylabel('Z (km)', fontsize=11)
		plt.yticks(fontsize=10)
		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="6%", pad=0.15)
		cbar = plt.colorbar(im, cax=cax)
		cbar.set_label('km/s',size=10)
		cbar.ax.tick_params(labelsize=10)


		plt.style.use('default')
		plt.figure(figsize=(5,9))
		plt.subplot(2,1,1)
		ax = plt.gca()
		im = ax.imshow(np.abs(v_hard.reshape(101,101)), vmin=2, vmax=3, extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
		plt.xlabel('X (km)', fontsize=11)
		plt.xticks(fontsize=10)
		plt.ylabel('Z (km)', fontsize=11)
		plt.yticks(fontsize=10)
		plt.title('mean', fontsize=15)
		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="6%", pad=0.15)
		cbar = plt.colorbar(im, cax=cax)
		cbar.set_label('km/s',size=10)
		cbar.ax.tick_params(labelsize=10)


		plt.subplot(2,1,2)
		ax = plt.gca()
		im = ax.imshow(np.sqrt(var_v.reshape(101,101)), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
		plt.xlabel('X (km)', fontsize=11)
		plt.xticks(fontsize=10)
		plt.ylabel('Z (km)', fontsize=11)
		plt.yticks(fontsize=10)
		plt.title('std', fontsize=15)
		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="6%", pad=0.15)
		cbar = plt.colorbar(im, cax=cax)
		cbar.set_label('km/s',size=10)
		cbar.ax.tick_params(labelsize=10)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
		    wspace=None, hspace=0.25)
		plt.savefig('inclu_svgd_0.8657_0.0606.png', bbox_inches='tight')

		plt.style.use('default')
		plt.figure(figsize=(5,5))
		ax = plt.gca()
		im = ax.imshow(np.abs(T_pred), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
		plt.xlabel('X (km)', fontsize=11)
		plt.xticks(fontsize=10)
		plt.ylabel('Z (km)', fontsize=11)
		plt.yticks(fontsize=10)
		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="6%", pad=0.15)
		cbar = plt.colorbar(im, cax=cax)
		cbar.set_label('s',size=10)
		cbar.ax.tick_params(labelsize=10)

		plt.style.use('default')
		plt.figure(figsize=(5,5))
		ax = plt.gca()
		im = ax.imshow(np.abs(T_data), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
		plt.xlabel('X (km)', fontsize=11)
		plt.xticks(fontsize=10)
		plt.ylabel('Z (km)', fontsize=11)
		plt.yticks(fontsize=10)
		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="6%", pad=0.15)
		cbar = plt.colorbar(im, cax=cax)
		cbar.set_label('s',size=10)
		cbar.ax.tick_params(labelsize=10)

		fig = plt.figure(figsize=(5,5))
		ax = plt.gca()
		im1 = ax.contour(T_data, 12, extent=[xmin,xmax,zmin,zmax], colors='r')
		im2 = ax.contour(T_pred, 12, extent=[xmin,xmax,zmin,zmax], colors='k',linestyles = 'dashed')
		ax.plot(sx,sz,'r*',markersize=12)

		plt.xlabel('X (km)', fontsize=11)
		plt.ylabel('Z (km)', fontsize=11)
		ax.tick_params(axis='both', which='major', labelsize=8)
		plt.gca().invert_yaxis()
		h1,_ = im1.legend_elements()
		h2,_ = im2.legend_elements()

		ax.legend([h1[0], h2[0]], ['FMM', 'BPINNs-SVGD'],fontsize=12)

		ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
		ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		plt.savefig('t_svgd_0.9979_0.0290.png', bbox_inches='tight')



		vt_diff = np.abs((v_hard.reshape(101,101)-velmodel)/velmodel).reshape(-1)
		diff = sum(vt_diff)*(1/len(velmodel.reshape(-1)))
		print('diff_sum_v',diff)


		t_diff = np.abs((T_pred - T_data) / T_data).reshape(-1)
		diff = sum(t_diff)*(1/len(T_data.reshape(-1)))
		print('diff_sum_t',diff)


		vt_pred = v_hard.reshape(-1)
		velmodel1d = velmodel.reshape(-1)
		x_mean = np.mean(vt_pred, axis=-1, keepdims=True)
		y_mean = np.mean(velmodel1d, axis=-1, keepdims=True)
		x_std = np.std(vt_pred, axis=-1, keepdims=True)
		y_std = np.std(velmodel1d, axis=-1, keepdims=True)
		corr = np.mean((vt_pred-x_mean)*(velmodel1d-y_mean), axis=-1,keepdims=True)/(x_std*y_std)
		print('corr_v',corr)

		t_pred = T_pred.reshape(-1)
		t_data = T_data.reshape(-1)
		x_mean = np.mean(t_pred, axis=-1, keepdims=True)
		y_mean = np.mean(t_data, axis=-1, keepdims=True)
		x_std = np.std(t_pred, axis=-1, keepdims=True)
		y_std = np.std(t_data, axis=-1, keepdims=True)
		corr = np.mean((t_pred-x_mean)*(t_data-y_mean), axis=-1,keepdims=True)/(x_std*y_std)
		print('corr_t',corr)


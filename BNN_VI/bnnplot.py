import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import torch
import skfmm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from mpl_toolkits.axes_grid1 import make_axes_locatable
from args import args, device

# Plot the output result
def plot(net, vmin, vmax):
    v0 = 2.; # Velocity at the origin of the model
    vergrad = 1; # Vertical gradient
    horgrad = 0.; # Horizontal gradient

    zmin = 0.; zmax = 2.; deltaz = 0.02;
    xmin = 0.; xmax = 2.; deltax = 0.02;

    z = np.arange(zmin, zmax + deltaz, deltaz)
    nz = z.size

    x = np.arange(xmin, xmax + deltax, deltax)
    nx = x.size

    Z,X = np.meshgrid(z,x,indexing='ij')


    velmodel = v0*np.ones_like(Z)

    for i in range(len(velmodel)):
        for j in range(len(velmodel[0])):
            if ((i*0.02-1)/0.4)**2 + ((j*0.02-1)/0.6)**2 <= 1:
                velmodel[i][j] = 3

    sz = 1; sx = 0
    samples = 100
    vmax = float(vmax);vmin = float(vmin)

    phi = -1*np.ones_like(velmodel)
    phi[np.logical_and(Z==sz, X==sx)] = 1
    d = skfmm.distance(phi, dx=2e-2)
    T_data = skfmm.travel_time(phi, velmodel, dx=2e-2)
    
    Z = torch.FloatTensor(Z.reshape(-1,1))
    X = torch.FloatTensor(X.reshape(-1,1))
    SZ = sz*torch.ones_like(Z)
    SX = sx*torch.ones_like(X)
    
    input_xsx = torch.cat((SZ,SX,Z,X),1).to(device)
    input_x = torch.cat((Z,X),1).to(device)
    
    output = net(input_xsx,input_x).detach()
    print('output',output.shape)


    
    v_samp = torch.zeros((input_xsx.size(0),samples))
    tau_samp = torch.zeros((input_xsx.size(0),samples))
    
    for s in range(samples):
        output = net(input_xsx,input_x).detach()
        tau_samp[:,s] = output[:,0].reshape(-1)
        v_samp[:,s] = output[:,1].reshape(-1)
        v_samp[:,s] = v_samp[:,s]*(vmax-vmin) + vmin
        
        
    v_hard = torch.mean(v_samp, axis = 1)
    tau_pred = torch.mean(tau_samp, axis = 1)
    v_var = torch.mean(v_samp.pow(2), axis = 1) - v_hard.pow(2)
    v_std = torch.sqrt(v_var+0.00001)
    vtrue = torch.FloatTensor(velmodel)
    


    vs = velmodel[int(sz/deltaz),int(sx/deltax)]
    T0 = np.sqrt((Z-sz)**2+(X-sx)**2)/vs
    T0 = T0.reshape(101, 101)
    T_pred = tau_pred.reshape(101, 101)*T0


    plt.style.use('default')
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(np.abs(vtrue.reshape(101,101)), extent=[xmin,xmax,zmax,zmin], vmin=2, vmax=3, aspect=1, cmap="jet")
    plt.xlabel('X (km)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('Z (km)', fontsize=12)
    plt.yticks(fontsize=10)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('km/s',size=10)
    cbar.ax.tick_params(labelsize=10)


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

    ax.legend([h1[0], h2[0]], ['FMM', 'BPINN-VI'],fontsize=12)

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('t_vi_0.9982_0.0344.png', bbox_inches='tight')
    

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
    im = ax.imshow(np.abs(v_std.reshape(101,101)), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
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
    plt.savefig('inclu_vi_0.8490_0.0586.png', bbox_inches='tight')

    v_std = v_std.view(-1, 1)
    print('v_std', max(v_std), v_std.mean())

    plt.style.use('default')
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(np.abs(T_pred), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
    plt.xlabel('X (km)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('Z (km)', fontsize=12)
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
    plt.xlabel('X (km)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('Z (km)', fontsize=12)
    plt.yticks(fontsize=10)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('s',size=10)
    cbar.ax.tick_params(labelsize=10)
    
    vt_diff = np.abs((v_hard.reshape(101,101)-velmodel)/velmodel).reshape(-1)
    diff = sum(vt_diff)*(1/len(velmodel.reshape(-1)))
    print('diff_sum',diff)


    vt_pred = v_hard.reshape(-1)
    velmodel1d = velmodel.reshape(-1)
    vt_pred = vt_pred.numpy()
    x_mean = np.mean(vt_pred, axis=-1, keepdims=True)
    y_mean = np.mean(velmodel1d, axis=-1, keepdims=True)
    x_std = np.std(vt_pred, axis=-1, keepdims=True)
    y_std = np.std(velmodel1d, axis=-1, keepdims=True)
    corr = np.mean((vt_pred-x_mean)*(velmodel1d-y_mean), axis=-1,keepdims=True)/(x_std*y_std)
    print('corr',corr)
    
    
    t_diff = np.abs((T_pred - T_data)/T_data).reshape(-1)
    diff = sum(t_diff)*(1/len(T_data.reshape(-1)))
    print('diff_sum',diff)


    t_pred = T_pred.reshape(-1)
    t_data = T_data.reshape(-1)
    t_pred = t_pred.numpy()

    x_mean = np.mean(t_pred, axis=-1, keepdims=True)
    y_mean = np.mean(t_data, axis=-1, keepdims=True)
    x_std = np.std(t_pred, axis=-1, keepdims=True)
    y_std = np.std(t_data, axis=-1, keepdims=True)
    corr = np.mean((t_pred-x_mean)*(t_data-y_mean), axis=-1,keepdims=True)/(x_std*y_std)
    print('corr',corr)
    
    
    vpred = v_hard.reshape(velmodel.shape)[51, :]
    vtrue = velmodel[51, :]
    std = v_std.reshape(velmodel.shape)[51, :]
    y1 = vpred - std
    y2 = vpred + std

    p0 = v_samp[:,0].detach().numpy().reshape(velmodel.shape)
    p1 = v_samp[:,20].detach().numpy().reshape(velmodel.shape)
    p2 = v_samp[:,40].detach().numpy().reshape(velmodel.shape)
    p3 = v_samp[:,60].detach().numpy().reshape(velmodel.shape)
    p4 = v_samp[:,80].detach().numpy().reshape(velmodel.shape)
    n = 5; x_ticks = np.arange(0, xmax+deltax, deltax)
    x = np.arange(0, xmax+deltax, deltax)

    plt.figure()
    # plt.scatter(x_ticks, p0[51,:][::n], c='b')
    # plt.scatter(x_ticks, p1[51,:][::n], c='b')
    # plt.scatter(x_ticks, p2[51,:][::n], c='b')
    # plt.scatter(x_ticks, p3[51,:][::n], c='b')
    # plt.scatter(x_ticks, p4[51,:][::n], c='b')
    plt.plot(x_ticks,p0[51, :],linestyle='-', color = 'g', lw=2.0, alpha=0.8)
    plt.plot(x_ticks,p1[51, :],linestyle='-', color = 'r', lw=2.0, alpha=0.8)
    #plt.plot(x_ticks,p2[51, :],linestyle='-', color = 'b', lw=2.0, alpha=0.8)
    # plt.plot(x,vtrue,linestyle='-', color = 'k', lw=2.0, alpha=1.0)
    plt.plot(x,vpred,linestyle='-', color = 'k', lw=2.0, alpha=1.0)
    plt.fill_between(x, y1,y2, where=(y1>y2)|(x>=0),alpha = 0.3, facecolor='blue')

    plt.xlabel('X(km)', fontsize=11)
    plt.ylabel('v(km/s)', fontsize=11)
    plt.show()

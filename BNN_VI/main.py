from args import args, device
import torch
import model
from loader import loader
import time
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import torch.nn as nn
from bnnplot import plot


epochs = args.epochs
train_size = args.batch_size
nsamples= args.nsamples

net = model.MLP_BBB().to(device)

unlabeled_data, labeled_data, labeled_data_s, vmin, vmax = loader().data_loader()

tic = time.time()

optimizer = optim.Adam([{'params':net.predictt.parameters(),'lr': 0.01},
                        {'params': net.hiddent.parameters(),'lr': 0.01},
                        {'params': net.predictv.parameters(),'lr': 0.01},
                        {'params': net.hiddenv.parameters(),'lr': 0.01},
                        {'params': net.featuret.parameters()},
                        {'params': net.featurev.parameters()},],lr =1e-3)

for epoch in range(epochs):
    net.train()
    for batch_idx, (zs, xs, z, x, vs) in enumerate(unlabeled_data):
        zs_r = labeled_data[:, 0].reshape(-1, 1)
        xs_r = labeled_data[:, 1].reshape(-1, 1)
        zr_r = labeled_data[:, 2].reshape(-1, 1)
        xr_r = labeled_data[:, 3].reshape(-1, 1)
        zr1 = labeled_data[:, 2].reshape(-1, 1)
        xr1 = labeled_data[:, 3].reshape(-1, 1)
        zs_r.requires_grad = True
        xs_r.requires_grad = True
        zr_r.requires_grad = True
        xr_r.requires_grad = True
        zr1.requires_grad = True
        xr1.requires_grad = True

        net_in_xsx = torch.cat((zs_r, xs_r, zr_r, xr_r), 1)
        net_in_x = torch.cat((zr1, xr1), 1)


        vref = labeled_data[:, -2].reshape(-1, 1)
        tauref = labeled_data[:, -1].reshape(-1, 1)

        v_noise = torch.normal(0, 0.05*vref)+vref
        tau_noise = torch.normal(0, 0.05*tauref)+tauref
        targetref = torch.cat((tauref, vref), 1)


        optimizer.zero_grad()

        lossd = net.sample_elbo(net_in_xsx, net_in_x, targetref, nsamples)
        
        T0 = np.sqrt((z - zs) ** 2 + (x - xs) ** 2) / vs
        px0 = np.divide(x - xs, T0 * vs ** 2, out=np.zeros_like(T0), where=T0 != 0)
        pz0 = np.divide(z - zs, T0 * vs ** 2, out=np.zeros_like(T0), where=T0 != 0)

        px0 = torch.FloatTensor(px0).to(device)
        pz0 = torch.FloatTensor(pz0).to(device)


        zs = torch.FloatTensor(zs).to(device)
        xs = torch.FloatTensor(xs).to(device)
        z1 = torch.FloatTensor(z).to(device)
        x1 = torch.FloatTensor(x).to(device)
        z2 = torch.FloatTensor(z).to(device)
        x2 = torch.FloatTensor(x).to(device)
        
        vs = torch.FloatTensor(vs).to(device)


        zs.requires_grad = True
        xs.requires_grad = True
        z1.requires_grad = True
        x1.requires_grad = True
        z2.requires_grad = True
        x2.requires_grad = True

        T0 = torch.sqrt((z1 - zs) ** 2 + (x1 - xs) ** 2).div(vs)


        rc_in_xsx = torch.cat((zs, xs, z1, x1), 1)
        rc_in_x = torch.cat((z2, x2), 1)

        outputr = net.forward(rc_in_xsx, rc_in_x)

        tau = outputr[:, 0]
        v = outputr[:, 1] * (vmax - vmin) + vmin

        tau = tau.view(len(tau), -1)
        v = v.view(len(tau), -1)
        T0 = T0.view(len(T0), -1)

        tau_x = torch.autograd.grad(tau, x1, grad_outputs=torch.ones_like(x1), create_graph=True, only_inputs=True)[0]
        tau_z = torch.autograd.grad(tau, z1, grad_outputs=torch.ones_like(x1), create_graph=True, only_inputs=True)[0]
        rpc = ((T0 * tau_x + tau * px0) ** 2 + (T0 * tau_z + tau * pz0) ** 2) - 1 /v ** 2

        # calculate the log likelihood of physics equation
        lossr = Normal(rpc.reshape(-1), 0.01).log_prob(torch.zeros_like(rpc.reshape(-1))).sum()
        loss = lossd.to(device) + (-lossr)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
                output = net.forward(net_in_xsx, net_in_x)
                loss_f = nn.MSELoss()
                loss_r = loss_f(rpc,torch.zeros_like(rpc))
                loss_tau = loss_f(output[:, 0], targetref[:, 0])
                loss_v = loss_f(output[:, 1], targetref[:, 1])
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvgLossR: {:.10f}\tAvgLosstau: {:.10f}\tAvgLossv: {:.10f}'.format(
                        epoch, batch_idx * len(x), len(unlabeled_data) * len(x),
                                100. * batch_idx / len(unlabeled_data), loss_r.item(), loss_tau.item(), loss_v.item()))
    if epoch%100==0:
        torch.save(net.state_dict(),"test"+str(epoch)+"_noise1.pt")


training_time = time.time() - tic
print('finished in ',training_time)

net.load_state_dict(torch.load("test800_noise1.pt",map_location = 'cpu'))
plot(net, vmin, vmax)
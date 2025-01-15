import numpy as np
import torch
import skfmm
from args import args,device
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

class loader():
    def data_loader(self):
        train_size = args.batch_size
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

        selected_pts1_1 = np.linspace(0, Z.size-101, num = 5).astype('int64')
        selected_pts1_2 = np.linspace(100, Z.size-1, num = 5).astype('int64')
        selected_pts2_1 = np.linspace(0, Z.size-101, num = 51).astype('int64')
        selected_pts2_2 = np.linspace(100, Z.size-1, num = 51).astype('int64')

        selected_pts1 = np.append(selected_pts1_1, selected_pts1_2)
        selected_pts2 = np.append(selected_pts2_1, selected_pts2_2)

        Z = Z.reshape(-1)
        X = X.reshape(-1)

        Zs = Z[selected_pts1]
        Xs = X[selected_pts1]

        Zr = Z[selected_pts2]
        Xr = X[selected_pts2]


        velmodel_min = np.min(velmodel, keepdims=True)
        velmodel_max = np.max(velmodel, keepdims=True)
        velmodel_nm = (velmodel - velmodel_min) / (velmodel_max - velmodel_min)

        # Extract unlabeled data to calculate the likelihood of physics equation
        Z_u, X_u, Zs_u, Xs_u, vs_u = [], [], [], [], []
        for ns, (szi, sxi) in enumerate(zip(Zs, Xs)):
            Zs_ui = szi * np.ones_like(Z)
            Xs_ui = sxi * np.ones_like(X)

            vs = velmodel[int(szi / deltaz), int(sxi / deltax)]
            vsi = vs * np.ones_like(X)
            Zs_u = np.concatenate((Zs_u, Zs_ui), 0)
            Xs_u = np.concatenate((Xs_u, Xs_ui), 0)
            Z_u = np.concatenate((Z_u, Z), 0)
            X_u = np.concatenate((X_u, X), 0)
            vs_u = np.concatenate((vs_u, vsi), 0)
        data = torch.utils.data.TensorDataset(torch.FloatTensor(Zs_u.reshape(-1, 1)),
                                              torch.FloatTensor(Xs_u.reshape(-1, 1)),
                                              torch.FloatTensor(Z_u.reshape(-1, 1)), torch.FloatTensor(X_u.reshape(-1, 1)),
                                              torch.FloatTensor(vs_u.reshape(-1, 1)))
        unlabeled_data = torch.utils.data.DataLoader(data, batch_size=train_size, shuffle=False)

        print('len(data is)', len(data))
        print('len(dataloader is)', len(unlabeled_data))


        # Extract labeled data to calculate the data likelihood
        Z, X = np.meshgrid(z, x, indexing='ij')
        Zr_l, Xr_l, Zs_l, Xs_l, Vref, tauref = [], [], [], [], [], []

        for ns, (szi, sxi) in enumerate(zip(Zs, Xs)):

            vs = velmodel[int(round(szi / deltaz)), int(round(sxi / deltax))]

            phi = -1*np.ones_like(X)
            phi[np.logical_and(Z==szi, X==sxi)] = 1
            d = skfmm.distance(phi, dx=2e-2)
            T_data = skfmm.travel_time(phi, velmodel, dx=2e-2)


            T_data = T_data.reshape(-1)
            T0 = np.sqrt((Z - szi) ** 2 + (X - sxi) ** 2) / vs
            T0 = T0.reshape(-1)
            tau = np.divide(T_data, T0, out=np.ones_like(T0), where=T0 != 0)
            tau_data = tau.reshape(-1)[selected_pts2]
            tauref = np.concatenate((tauref, tau_data), 0)

            Vr = velmodel_nm.reshape(-1)[selected_pts2]

            zstmp = szi * np.ones_like(Zr)
            xstmp = sxi * np.ones_like(Xr)
            Zs_l = np.concatenate((Zs_l, zstmp), 0)
            Xs_l = np.concatenate((Xs_l, xstmp), 0)

            Zr_l = np.concatenate((Zr_l, Zr), 0)
            Xr_l = np.concatenate((Xr_l, Xr), 0)

            Vref = np.concatenate((Vref, Vr), 0)

        labeled_data = torch.cat((torch.FloatTensor(Zs_l.reshape(-1, 1)),
                                              torch.FloatTensor(Xs_l.reshape(-1, 1)),
                                              torch.FloatTensor(Zr_l.reshape(-1, 1)),
                                              torch.FloatTensor(Xr_l.reshape(-1, 1)),
                                              torch.FloatTensor(Vref.reshape(-1, 1)),
                                              torch.FloatTensor(tauref.reshape(-1, 1))), 1).to(device)

        V_s = velmodel_nm.reshape(-1)[selected_pts1]
        labeled_data_s = torch.cat((torch.FloatTensor(Zs.reshape(-1, 1)), torch.FloatTensor(Xs.reshape(-1, 1)),
                                    torch.FloatTensor(V_s.reshape(-1, 1))), 1).to(device)

        return unlabeled_data, labeled_data, labeled_data_s, torch.FloatTensor(velmodel_min).to(device), torch.FloatTensor(velmodel_max).to(device)

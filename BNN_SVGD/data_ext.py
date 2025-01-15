import numpy as np
import skfmm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


v0 = 2.;


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

X_star = [Z.reshape(-1,1), X.reshape(-1,1)] # Grid points for prediction 


selected_pts1_1 = np.linspace(0, Z.size-101, num = 5).astype('int64')
selected_pts1_2 = np.linspace(100, Z.size-1, num = 5).astype('int64')
selected_pts2_1 = np.linspace(0, Z.size-101, num = 51).astype('int64')
selected_pts2_2 = np.linspace(100, Z.size-1, num = 51).astype('int64')

selected_pts1 = np.append(selected_pts1_1, selected_pts1_2) # source points
selected_pts2 = np.append(selected_pts2_1, selected_pts2_2) # recieve points


Zs = Z.reshape(-1)[selected_pts1]
Xs = X.reshape(-1)[selected_pts1]
Zr = Z.reshape(-1)[selected_pts2]
Xr = X.reshape(-1)[selected_pts2]




plt.style.use('default')
plt.figure(figsize=(4,4))
ax = plt.gca()
im = ax.imshow(np.abs(velmodel), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
ax.plot(Xr,Zr,'r*', markersize=8)
ax.plot(Xs,Zs,'k*', markersize=12)
plt.xlabel('X (km)', fontsize=14)
plt.xticks(fontsize=10)
plt.ylabel('Z (km)', fontsize=14)
plt.yticks(fontsize=10)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('vel(km/s)',size=10)
cbar.ax.tick_params(labelsize=10)

Zrr,Xrr,Zsr,Xsr,Vs,Vref,Tref = [],[],[],[],[],[],[] # labeled data extract
taurr = []

velmodel_min = np.min(velmodel, keepdims=True)
print(velmodel_min)
velmodel_max = np.max(velmodel,keepdims=True)
print(velmodel_max)
velmodel_nm = (velmodel-velmodel_min)/(velmodel_max-velmodel_min) # normalize velmodel

V_s = velmodel_nm.reshape(-1)[selected_pts1]

for ns, (szi, sxi) in enumerate(zip(Zs, Xs)):

    vs = velmodel[int(round(szi/deltaz)),int(round(sxi/deltax))]

    phi = -1*np.ones_like(X)
    phi[np.logical_and(Z==szi, X==sxi)] = 1
    d = skfmm.distance(phi, dx=2e-2)
    T_data = skfmm.travel_time(phi, velmodel, dx=2e-2)
    T_data = T_data.reshape(-1)
    
    # for i in range(len(T_data)):
    #     T_data[i] += np.random.normal(0,0.1*np.abs(T_data[i]),1)

    T0 = np.sqrt((Z-szi)**2 + (X-sxi)**2)/vs
    T0 = T0.reshape(-1)
    tau = np.divide(T_data, T0, out=np.ones_like(T0), where=T0!=0)
    tau_data = tau.reshape(-1)[selected_pts2]
    taurr =  np.concatenate((taurr,tau_data),0)

    Tr = T_data.reshape(-1)[selected_pts2]

    Vr = velmodel_nm.reshape(-1)[selected_pts2]
    Zrr = np.concatenate((Zrr,Zr),0)
    Xrr = np.concatenate((Xrr,Xr),0)
    zstmp = szi*np.ones_like(Zr)
    xstmp = sxi*np.ones_like(Xr)

    vs = vs*np.ones_like(Zr)
    Zsr = np.concatenate((Zsr,zstmp),0)
    Xsr = np.concatenate((Xsr,xstmp),0)
    Vs = np.concatenate((Vs,vs),0)
    Vref = np.concatenate((Vref,Vr),0)
    Tref = np.concatenate((Tref,Tr),0)


T0 = np.sqrt((Zrr-Zsr)**2 + (Xrr-Xsr)**2)/Vs

np.savez('data_XrX',Xs_a=Xsr.reshape(-1,1),Zs_a=Zsr.reshape(-1,1),Xr_a=Xrr.reshape(-1,1),Zr_a = Zrr.reshape(-1,1)) 
np.savez('targetref',tauref = taurr.reshape(-1,1),Vref = Vref.reshape(-1,1),T0 = T0.reshape(-1,1))
np.savez('data_XsX',Xs=Xs.reshape(-1,1),Zs=Zs.reshape(-1,1),Vs=V_s)
np.savez('normalize',velmodel_min=velmodel_min,velmodel_max=velmodel_max)

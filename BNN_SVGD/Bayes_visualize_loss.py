import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


lossr = np.loadtxt('LOSSr.csv')
lossv = np.loadtxt('LOSSv.csv')
losstau = np.loadtxt('LOSStau.csv')


lossrt = lossr[:2000]
lossvt = lossv[:2000]
losstaut = losstau[:2000]


lossr = []
for i in range(0, 2000, 2):
    lossr.append((lossrt[i]+lossrt[i+1])/2)
lossr = np.array(lossr)

lossv = []
for i in range(0, 2000, 2):
    lossv.append((lossvt[i]+lossvt[i+1])/2)
lossv = np.array(lossv)
losstau = []
for i in range(0, 2000, 2):
    losstau.append((losstaut[i]+losstaut[i+1])/2)
losstau = np.array(losstau)



x_ticks = np.arange(0,1000)
x = np.arange(0,500)


lossr1 = []; lossv1 = []; losstau1 = []

for i in range(0,1000):
    if i%2 == 0:
        lossr1.append(lossr[i//2])
        lossv1.append(lossv[i//2])
        losstau1.append(losstau[i//2])
    else:
        lossr1.append((lossr[i//2]+lossr[i//2+1])/2)
        lossv1.append((lossv[i//2]+lossv[i//2+1])/2)
        losstau1.append((losstau[i//2]+losstau[i//2+1])/2)
lossr1 = np.array(lossr1)
lossv1 = np.array(lossv1)
losstau1 = np.array(losstau1)

# plt.figure()
# plt.rc('font',family='Times New Roman')
# plt.plot(x_ticks,lossr1+lossv1+losstau1, label="Loss sum")
# plt.plot(x_ticks,lossr1, label="Loss R")
# plt.plot(x_ticks,lossv1, label="Loss v")
# plt.plot(x_ticks,losstau1, label="Loss tau")
# plt.xticks(fontsize=12) 
# plt.yticks(fontsize=12)
# plt.yscale('log')
# font = {'family' : 'Times New Roman',
# 'size'   : 16,
# }
# plt.xlabel("Training Epochs",font)
# plt.ylabel("Mean Squared Test Error",font)
# plt.legend(fontsize=12)


plt.figure()
plt.plot(x_ticks,lossr1+lossv1+losstau1, label="Loss sum")
plt.plot(x_ticks,lossr1, label="Loss R")
plt.plot(x_ticks,lossv1, label="Loss v")
plt.plot(x_ticks,losstau1, label="Loss tau")
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
plt.yscale('log')
font = {'size'   : 16,
}
plt.xlabel("Training Epochs",font)
plt.ylabel("Mean Squared Test Error",font)
plt.legend(fontsize=12)


plt.figure()
plt.plot(x_ticks,lossr+lossv+losstau, label="Loss sum")
plt.plot(x_ticks,lossr, label="Loss R")
plt.plot(x_ticks,lossv, label="Loss v")
plt.plot(x_ticks,losstau, label="Loss tau")
plt.yscale('log')
plt.xlabel("Training epochs")
plt.ylabel("Dimensionless test error")
#plt.xlim(0, 1000)
plt.legend()


# plt.figure()
# plt.plot(x_ticks,lossr,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Lossr')
# plt.plot(x_ticks,lossv,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Lossv')
# plt.plot(x_ticks,losstau,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Losstau')
# plt.xlabel('Training epochs')
# plt.ylabel('Dimensionless test error')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5))
# plt.yscale('log')
# plt.legend(prop={'size': 9})


# plt.figure()
# plt.plot(x_ticks,lossr+lossv+losstau,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Loss')
# plt.xlabel('Training epochs')
# plt.ylabel('Dimensionless test error')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5))
# plt.yscale('log')
# #plt.yticks([1e-6,1e-5,1e-4])
# plt.legend(prop={'size': 9})
# plt.show()
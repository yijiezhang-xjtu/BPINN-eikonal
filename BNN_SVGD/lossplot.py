import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

loss1 = np.loadtxt('LOSS1.csv')
lossD = np.loadtxt('LOSSD.csv')
lossB = np.loadtxt('LOSSB.csv')

plt.figure()
plt.plot(loss1,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Residual')
plt.plot(lossD,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'LossD')
plt.plot(lossB,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'LossB')
plt.xlabel('Number of training collocation points')
plt.ylabel('test error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5))
plt.yscale('log')
#plt.yticks([1e-6,1e-5,1e-4])
plt.legend(prop={'size': 9})
plt.show()

plt.figure()
plt.plot(lossB+lossD+loss1,linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0,label = 'Loss')
plt.xlabel('Number of training collocation points')
plt.ylabel('Dimensionless test error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5))
plt.yscale('log')
#plt.yticks([1e-6,1e-5,1e-4])
plt.legend(prop={'size': 9})
plt.show()
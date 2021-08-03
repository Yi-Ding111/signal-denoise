import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


port_num=16
t=np.arange(2500)


'''
signal1=pd.read_csv('./f1y01_10s_interval.csv')
f1y01=list(signal1['0_10'])
f1y02=list(signal1['10_20'])
f1y03=list(signal1['20_30'])
f1y04=list(signal1['30_40'])
f1y05=list(signal1['40_50'])
f1y06=list(signal1['50_60'])
f1y07=list(signal1['60_70'])
f1y08=list(signal1['70_80'])
f1y09=list(signal1['80_90'])
f1y10=list(signal1['90_100'])
'''


signal1=pd.read_csv('/Users/charles/Desktop/微波降噪/ECG/f1y01_f1y10_ECG.csv')

f1y01=list(signal1['f1y01'])
f1y02=list(signal1['f1y02'])
f1y03=list(signal1['f1y03'])
f1y04=list(signal1['f1y04'])
f1y05=list(signal1['f1y05'])
f1y06=list(signal1['f1y06'])
f1y07=list(signal1['f1y07'])
f1y08=list(signal1['f1y08'])
f1y09=list(signal1['f1y09'])
f1y10=list(signal1['f1y10'])


#mix all signals
#df_list=['0_10','10_20','20_30','30_40','40_50','50_60','60_70','70_80','80_90','90_100']
df_list=['f1y01','f1y02','f1y03','f1y04','f1y05','f1y06','f1y07','f1y08','f1y09','f1y10']
mix_matrix=[]

for i in df_list:
    a=signal1[i]
    mix_matrix.append(a)

mix_matrix=np.array(mix_matrix)
#print(np.shape(mix_matrix))
#implement fastICA
ica=FastICA(n_components=4)
mix_matrix=mix_matrix.T
u=ica.fit_transform(mix_matrix)
u=u.T

#draw the top 4 ICA components
x = np.arange(2500)
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
ax1.plot(x,u[0,:])
ax2.plot(x,u[1,:])
ax3.plot(x,u[2,:])
ax4.plot(x,u[3,:])
plt.show()


noise=u[2,:]
ECG_signal=f1y01
denoise=[ECG_signal[i]-noise[i] for i in range(0,len(noise))]


plt.subplot(3,1,1)
plt.plot(t,ECG_signal)
plt.subplot(3,1,2)
plt.plot(t,noise)
plt.subplot(3,1,3)
plt.plot(t,denoise)
plt.show()



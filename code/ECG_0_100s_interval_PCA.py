
from pandas.core import indexing
from pandas.core.algorithms import mode
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA

t=np.arange(2500)



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


plt.subplot(5,2,1)
plt.plot(t,f1y01)
plt.title('0_10s')
plt.subplot(5,2,2)
plt.plot(t,f1y02)
plt.title('10_20s')
plt.subplot(5,2,3)
plt.plot(t,f1y03)
plt.title('20_30s')
plt.subplot(5,2,4)
plt.plot(t,f1y04)
plt.title('30_40s')
plt.subplot(5,2,5)
plt.plot(t,f1y05)
plt.title('40_50s')
plt.subplot(5,2,6)
plt.plot(t,f1y06)
plt.title('50_60s')
plt.subplot(5,2,7)
plt.plot(t,f1y07)
plt.title('60_70s')
plt.subplot(5,2,8)
plt.plot(t,f1y08)
plt.title('70_80s')
plt.subplot(5,2,9)
plt.plot(t,f1y09)
plt.title('80_90s')
plt.subplot(5,2,10)
plt.plot(t,f1y10)
plt.title('90_100s')
plt.show()



#alignment
a=f1y02
b=f1y03
'''
plt.plot(t,a,'r',label='x')
plt.plot(t,b,'b',label='y')
plt.show()
'''
#extract the first peak of signal
#a_front=a[0:250]
#b_front=b[0:250]

#do signal alignment
#c12=np.correlate(a_front,b_front,mode='full')
#t12=np.argmax(c12)
#len_a_f=len(a_front)
#index=t12-len_a_f
#if index>0:
    #tt1=b_front[index:]
    #tt2=b_front[0:index]
    #b_0=tt1+tt2
#else:
    #index=len_a_f+index
    #tt1=b_front[0:index]
    #tt2=b_front[index:]
    #b_0=tt2+tt1

'''
index=22
tt1=b_front[index:]
tt2=b_front[0:index]
b_0=tt1+tt2

plt.subplot(2,1,1)
plt.plot(range(len(a_front)),a_front,'r',label='x')
plt.plot(range(len(a_front)),b_front,'b',label='y')
plt.subplot(2,1,2)
plt.plot(range(len(a_front)),a_front,'r',label='x')
plt.plot(range(len(a_front)),b_0,'b',label='y')
plt.show()


c23=np.correlate(a,b,mode='same')
t23=np.argmax(c23)

len_a=len(a)
index=t23-len_a
if index>0:
    tt1=b[index:]
    tt2=b[0:index]
    b_0=tt1+tt2
else:
    index=len_a+index
    tt1=b[0:index]
    tt2=b[index:]
    b_0=tt2+tt1
'''

#align signals manually
index=21
tt1=b[index:]
tt2=b[0:index]
b_0=tt1+tt2

plt.subplot(2,1,1)
plt.plot(t,a,'r',label='10-20s')
plt.plot(t,b,'b',label='20-30s')
plt.legend()
plt.title('before alignment')
plt.subplot(2,1,2)
plt.plot(t,a,'r',label='10-20s')
plt.plot(t,b_0,'b',label='20-30s')
plt.legend()
plt.title('after alignment')
plt.show()


#intercept signals
a1=a[:2300]
b1=b_0[:2300]
dict_2={'a':a1,'b':b1}
df_2=pd.DataFrame(dict_2)


class Signal_PCA:

    def __init__(self,n_components):
        self.n_components=n_components
        self.mean_val=None
        self.new_data=None
        self.n_max_eig_vects=None

    def fit(self,data):
        '''
        Find eigenvertors and eigenvalues
        '''
        self.mean_val=np.mean(data,axis=0)
        self.new_data=data-self.mean_val
        #covariance matrix
        #rowvar=0 means each row represent one sample, one column is one feature
        cov_matrix=np.cov(self.new_data,rowvar=0)
        #eigen value and eigen vertors
        eig_vals,eig_vects=np.linalg.eig(np.mat(cov_matrix))
        #sort the eigenvalues
        eig_vals_sort=np.argsort(eig_vals)
        #find maximum eigenvalue and eigenvertors
        n_eig_vals_sort=eig_vals_sort[-1:-(self.n_components+1):-1]
        self.n_max_eig_vects=eig_vects[:,n_eig_vals_sort]


    def SVD(self,data):
        self.mean_val_2=np.mean(data,axis=0)
        self.new_data_2=data-self.mean_val_2
        cov_mat = np.dot(self.new_data_2.T, self.new_data_2)
        U, s, V = np.linalg.svd(cov_mat) 
        pc = np.dot(self.new_data_2, U) 
        return pc

    def transform(self):
        '''
        get dimensionality reduction data
        '''
        new_data=self.new_data.values
        return np.dot(new_data,self.n_max_eig_vects)



#PCA
ecg_pca=Signal_PCA(10)
ecg_pca.fit(signal1)
inter_signal=ecg_pca.transform()
inter_signal_arr_scattering=(inter_signal.T).A

'''
print(inter_signal_arr_scattering[0])
plt.plot(t,inter_signal_arr_scattering[0])
plt.show()
'''

noise=inter_signal_arr_scattering[2]
ECG_signal=f1y10
denoise=[ECG_signal[i]-noise[i] for i in range(0,len(noise))]

plt.subplot(3,1,1)
plt.plot(t,ECG_signal)
plt.legend()
plt.title('original_90-100s')
plt.subplot(3,1,2)
plt.plot(t,noise)
plt.legend()
plt.title('noise')
plt.subplot(3,1,3)
plt.plot(t,denoise)
plt.legend()
plt.title('signal after dropping noise')
plt.show()


ecg_pca=Signal_PCA(2)
ecg_pca.fit(df_2)
inter_signal=ecg_pca.transform()
inter_signal_arr_scattering=(inter_signal.T).A
noise=inter_signal_arr_scattering[1]
ECG_signal=a1
denoise=[ECG_signal[i]-noise[i] for i in range(0,len(noise))]

plt.subplot(3,1,1)
plt.plot(range(len(a1)),ECG_signal)
plt.legend()
plt.title('original_0-10s(2300)')
plt.subplot(3,1,2)
plt.plot(range(len(a1)),noise)
plt.legend()
plt.title('noise')
plt.subplot(3,1,3)
plt.plot(range(len(a1)),denoise)
plt.legend()
plt.title('0-10s(2300) drop noise')
plt.show()


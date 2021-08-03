
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA

t=np.arange(2500)


signal1=pd.read_csv('./f1y01_f1y10_ECG.csv')

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


plt.subplot(5,2,1)
plt.plot(t,f1y01)
plt.title('f1y01')
plt.subplot(5,2,2)
plt.plot(t,f1y02)
plt.title('f1y02')
plt.subplot(5,2,3)
plt.plot(t,f1y03)
plt.title('f1y03')
plt.subplot(5,2,4)
plt.plot(t,f1y04)
plt.title('f1y04')
plt.subplot(5,2,5)
plt.plot(t,f1y05)
plt.title('f1y05')
plt.subplot(5,2,6)
plt.plot(t,f1y06)
plt.title('f1y06')
plt.subplot(5,2,7)
plt.plot(t,f1y07)
plt.title('f1y07')
plt.subplot(5,2,8)
plt.plot(t,f1y08)
plt.title('f1y08')
plt.subplot(5,2,9)
plt.plot(t,f1y09)
plt.title('f1y09')
plt.subplot(5,2,10)
plt.plot(t,f1y10)
plt.title('f1y10')
plt.show()




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
        #singular value decomposition of convariance matrix
        U, s, V = np.linalg.svd(cov_mat)
        #return the reduced dimensionality matrix
        pc = np.dot(self.new_data_2, U) 
        return pc#pc[:,97]

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
inter_signal_arr_scattering=-(inter_signal.T).A

'''
print(inter_signal_arr_scattering[0])
plt.plot(t,inter_signal_arr_scattering[0])
plt.show()
'''

noise=inter_signal_arr_scattering[1]
ECG_signal=f1y05
denoise=[ECG_signal[i]-noise[i] for i in range(0,len(noise))]

plt.subplot(3,1,1)
plt.plot(t,ECG_signal,label='signal:f1y05')
plt.legend()
plt.title('original')
plt.subplot(3,1,2)
plt.plot(t,noise,label='PCA_component')
plt.legend()
plt.title('PCA_component')
plt.subplot(3,1,3)
plt.plot(t,denoise,label='signal_PCA')
plt.legend()
plt.title('after PCA')
plt.show()






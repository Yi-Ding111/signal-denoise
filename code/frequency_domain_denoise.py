import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA

port_num=16
t=np.arange(751)


ntwk=rf.Network('./dscan1-emscan-e00-cal.s16p')
ntwk2=rf.Network('./dscan5-emscan-e00-cal.s16p')


def freq_domain_signal(ntwk):
    '''
    read signals into dictionary
    '''
    fd_signal=locals()  
    for i in range(0,port_num):
        for j in range(0,port_num):
            fd_signal['fd'+str(i+1)+','+str(j+1)]=ntwk.s[:,i,j]
    fd_signal.pop('ntwk')
    return fd_signal


#classify signals
class_1=['fd1,2','fd2,3','fd3,4','fd4,5','fd5,6','fd6,7','fd7,8','fd8,9','fd9,10','fd10,11','fd11,12','fd12,13','fd13,14','fd14,15','fd15,16','fd16,1']
class_2=['fd1,3','fd2,4','fd3,5','fd4,6','fd5,7','fd6,8','fd7,9','fd8,10','fd9,11','fd10,12','fd11,13','fd12,14','fd13,15','fd14,16','fd15,1','fd16,2']
class_3=['fd1,4','fd2,5','fd3,6','fd4,7','fd5,8','fd6,9','fd7,10','fd8,11','fd9,12','fd10,13','fd11,14','fd12,15','fd13,16','fd14,1','fd15,2','fd16,3']

fd_signal=freq_domain_signal(ntwk)
fd_signal_1={key: value for key, value in fd_signal.items() if key in class_1}
fd_signal_1={key: value for key, value in fd_signal.items() if key in class_1}
fd_signal_1={key: value for key, value in fd_signal.items() if key in class_1}


#draw part of signals of class_1
a=fd_signal_1
plt.subplot(3,1,1)
td12=a['fd1,2']
plt.plot(t,td12,label='s1,2')
plt.legend()
plt.title('S-1,2')
plt.subplot(3,1,2)
td23=a['fd2,3']
plt.plot(t,td23,label='s2,3')
plt.legend()
plt.title('s-2,3')
plt.subplot(3,1,3)
td34=a['fd3,4']
plt.plot(t,td34,label='s3,4')
plt.legend()
plt.title('s-3,4')
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
        return pc#pc[:,2]

    def transform(self):
        '''
        get dimensionality reduction data
        '''
        new_data=self.new_data.values
        return np.dot(new_data,self.n_max_eig_vects)



def interference_signal_scatter():
    '''
    get the interference signal of scattering
    '''

    fd_signal_df=pd.DataFrame(fd_signal_1)
    signal_pca=Signal_PCA(5)
    #signal_pca.fit(fd_signal_df)
    #inter_signal=signal_pca.transform()
    inter_signal=(signal_pca.SVD(fd_signal_df))
    #inter_signal_arr_scattering=-(inter_signal.T).A

    #return inter_signal_arr_scattering
    return inter_signal


'''
plt.subplot(2,1,1)
plt.plot(t,ntwk.s[:,1,3])
plt.subplot(2,1,2)
#plot catter PCA
#inter_signal_arr_scattering=np.array(interference_signal_scatter())[2]
inter_signal_arr_scattering=np.array(interference_signal_scatter())
plt.plot(t,inter_signal_arr_scattering)
plt.show()
'''


def fd_signal_drop_pca_scatter():

    inter_signal_arr_scattering=interference_signal_scatter()
    td_values=list(fd_signal_1.values())
    td_signal_keys=list(fd_signal_1.keys())
    fd_signal_value=[]

    for i in range(len(td_values)):
        #td_signal_value.append((np.array(td_values[i])-np.array(inter_signal_arr_scattering))[4])
        fd_signal_value.append((np.array(td_values[i])+np.array(inter_signal_arr_scattering[:,5])))
    
    td_signal_scatter=dict(zip(td_signal_keys,fd_signal_value))
    
    return td_signal_scatter



plt.subplot(3,1,1)
a=fd_signal_1
fd12=a['fd2,3']
plt.plot(t,fd12,label='s2,3')
plt.legend()
plt.title('original')
plt.subplot(3,1,2)
c=np.array(interference_signal_scatter()[:,5])
plt.plot(t,c,label='SVD_component')
plt.legend()
plt.title('SVD_component')
plt.subplot(3,1,3)
b=fd_signal_drop_pca_scatter()
fd12_2=b['fd2,3']
print(len(fd12_2))
plt.plot(t,fd12_2,label='s2,3SVD')
plt.legend()
plt.title('after SVD')
plt.show()



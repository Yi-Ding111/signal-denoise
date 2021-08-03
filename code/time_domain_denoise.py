import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA

port_num=16


ntwk=rf.Network('./dscan1-emscan-e00-cal.s16p')
ntwk2=rf.Network('./dscan5-emscan-e00-cal.s16p')


def freq_domain_to_time_domain(ntwk):
    '''
    tranform frequency domain signal to time domain signal by IFFT
    '''
    td_signal=locals()
    #get the array of frequency
    freq=ntwk.f
    #get the step of between two frequencies
    delta_freq=freq[1]-freq[0]
    num_zero_freq=len(np.arange(0,freq[0],delta_freq))
    #create zero frequency array
    zero_freq_array=np.zeros(num_zero_freq)
    
    for i in range(0,port_num):
        for j in range(0,port_num):
            #put zero frequency and Sij together
            freq_half=np.hstack((zero_freq_array,ntwk.s[:,i,j]))
            #do conjugation and reversation to get the other half
            conj_reverse_freq_half=(((np.delete(freq_half,0)).conjugate()))[::-1]
            freq_full=np.hstack((freq_half,conj_reverse_freq_half))
            #time domain
            td_signal['td'+str(i+1)+','+str(j+1)]=np.fft.ifft(freq_full)*len(ntwk.s[:,i,j])
    td_signal.pop('ntwk')
    return td_signal


def signal_time(ntwk,N):
    '''
    calculate the time period
    '''
    freq=ntwk.f
    delta_freq=freq[1]-freq[0]
    time=1/delta_freq
    t=np.arange(0,time,time/N)
    return t



class Signal_PCA:
    '''
    '''

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


#classify scattering signals
class_1=['td1,2','td2,3','td3,4','td4,5','td5,6','td6,7','td7,8','td8,9','td9,10','td10,11','td11,12','td12,13','td13,14','td14,15','td15,16','td16,1']
class_2=['td1,3','td2,4','td3,5','td4,6','td5,7','td6,8','td7,9','td8,10','td9,11','td10,12','td11,13','td12,14','td13,15','td14,16','td15,1','td16,2']
class_3=['td1,4','td2,5','td3,6','td4,7','td5,8','td6,9','td7,10','td8,11','td9,12','td10,13','td11,14','td12,15','td13,16','td14,1','td15,2','td16,3']

td_signal=freq_domain_to_time_domain(ntwk)
td_signal_1={key: value for key, value in td_signal.items() if key in class_1}
td_signal_2={key: value for key, value in td_signal.items() if key in class_2}
td_signal_3={key: value for key, value in td_signal.items() if key in class_3}


#draw part of signals of class_1
a=td_signal_1
t=signal_time(ntwk,2001)
plt.subplot(3,1,1)
td12=a['td1,2']
plt.plot(t,td12,label='s1,2')
plt.legend()
plt.title('S-1,2')
plt.subplot(3,1,2)
td23=a['td2,3']
plt.plot(t,td23,label='s2,3')
plt.legend()
plt.title('s-2,3')
plt.subplot(3,1,3)
td34=a['td3,4']
plt.plot(t,td34,label='s3,4')
plt.legend()
plt.title('s-3,4')
plt.show()


def interference_signal_scatter():
    '''
    get the interference signal of scattering
    '''

    td_signal_df=pd.DataFrame(td_signal_1)
    #extract the top 5 principal components
    signal_pca=Signal_PCA(5)
    signal_pca.fit(td_signal_df)
    inter_signal=signal_pca.transform()
    #inter_signal=(signal_pca.SVD(fd_signal_df))
    inter_signal_arr_scattering=-(inter_signal.T).A
    return inter_signal_arr_scattering
    #return inter_signal


def td_signal_drop_pca_scatter():

    inter_signal_arr_scattering=interference_signal_scatter()
    td_values=list(td_signal_1.values())
    td_signal_keys=list(td_signal_1.keys())
    td_signal_value=[]

    for i in range(len(td_values)):
        td_signal_value.append((np.array(td_values[i])-np.array(inter_signal_arr_scattering))[2])
        #td_signal_value.append((np.array(td_values[i])-np.array(inter_signal_arr_scattering[:,2])))

    td_signal_scatter=dict(zip(td_signal_keys,td_signal_value))
    
    return td_signal_scatter


plt.subplot(3,1,1)
a=td_signal_1
td12=a['td1,2']
plt.plot(t,td12,label='s1,2')
plt.legend()
plt.title('original')
plt.subplot(3,1,2)
c=np.array(interference_signal_scatter()[2])
plt.plot(t,c,label='PCA_component')
plt.legend()
plt.title('PCA_component')
plt.subplot(3,1,3)
b=td_signal_drop_pca_scatter()
td12_2=b['td1,2']
print(len(td12_2))
plt.plot(t,td12_2,label='s1,2PCA')
plt.legend()
plt.title('after PCA')
plt.show()


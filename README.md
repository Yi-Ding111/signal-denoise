# signal-denoise
microwave/ECG signal denoise PCA


#PROJECT NAME 

microwave or ECG signal denoising through PCA/ICA/SVD

#DESCRIPTION 

the datasets from FANTASIA DATABASE

f1y01_f1y10_ECG.csv : including ten ECG records of ten people (f1y01 to f1y10), each record is 10 sec and 2500 samples.

f1y01_10s_interval.csv : including ten intervals of one people (f1y01), each record is 10 sec and 2500 samples.

the code have five parts:

ICA_ECG.py : using fast_ICA to denoise ECG signals

ECG_0_100s_interval_PCA.py : using PCA to denoise ten ECG signals from same person

ECG_10_signals_PCA.py : using PCA to denoise ten ECG signals from ten different people

frequency_domain_denoise.py : using PCA to denoise microwave signals in frequency domain

time_domain_denoise.py : using PCA to denoise microwave signals in time domain

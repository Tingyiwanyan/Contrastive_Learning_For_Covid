from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


k = np.load('7_data_icu_retain_ce_patient_data.npy')
logit = np.load('test_logit_icu.npy')

length = k.shape[0]
x_embed =TSNE(n_components=2).fit_transform(k)


for i in range(length):
    if logit[i,0] == 1:
        plt.plot(x_embed[i][0],x_embed[i][1],'.',color='red',markersize=6)
    if logit[i,1] == 1:
        plt.plot(x_embed[i][0],x_embed[i][1],'.',color='blue',markersize=10)

plt.show()
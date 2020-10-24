from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


k = np.load('7_data_mortality_retain_ce_patient_data.npy')
logit = np.load('test_logit_mortality.npy')

length = k.shape[0]
x_embed =TSNE(n_components=2).fit_transform(k)

fig, axs = plt.subplots(2,2)
"""
for i in range(length):
    if logit[i,0] == 1:
        axs[0,0].plot(x_embed[i][0],x_embed[i][1],'.',color='red',markersize=6)
    if logit[i,1] == 1:
        axs[0,0].plot(x_embed[i][0],x_embed[i][1],'.',color='blue',markersize=10)

for i in range(length):
    if logit[i,0] == 1:
        axs[0,1].plot(x_embed[i][0],x_embed[i][1],'.',color='red',markersize=6)
    if logit[i,1] == 1:
        axs[0,1].plot(x_embed[i][0],x_embed[i][1],'.',color='blue',markersize=10)

for i in range(length):
    if logit[i,0] == 1:
        axs[1,0].plot(x_embed[i][0],x_embed[i][1],'.',color='red',markersize=6)
    if logit[i,1] == 1:
        axs[1,0].plot(x_embed[i][0],x_embed[i][1],'.',color='blue',markersize=10)

for i in range(length):
    if logit[i,0] == 1:
        axs[1,1].plot(x_embed[i][0],x_embed[i][1],'.',color='red',markersize=6)
    if logit[i,1] == 1:
        axs[1,1].plot(x_embed[i][0],x_embed[i][1],'.',color='blue',markersize=10)

plt.show()
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


mortality_ce_7 = np.load('7_data_mortality_ce_24h.npy')
mortality_cl_7 = np.load('7_data_mortality_cl_24h.npy')
mortality_ce_23 = np.load('23_data_mortality_ce_24h.npy')
mortality_cl_23 = np.load('23_data_mortality_cl_24h.npy')
logit_7 = np.load('7_data_mortality_test_logit.npy')
logit_23 = np.load('23_data_mortality_test_logit.npy')

length_23 = mortality_ce_23.shape[0]
length_7 = mortality_ce_7.shape[0]
x_embed_ce_7 = TSNE(n_components=2).fit_transform(mortality_ce_7)
x_embed_cl_7 = TSNE(n_components=2).fit_transform(mortality_cl_7)
x_embed_ce_23 = TSNE(n_components=2).fit_transform(mortality_ce_23)
x_embed_cl_23 = TSNE(n_components=2).fit_transform(mortality_cl_23)

fig, axs = plt.subplots(2,2)

fig.suptitle('Mortality Prediction Embeddings')

for i in range(length_23):
    if logit_23[i,0] == 1:
        axs[0,0].plot(x_embed_cl_23[i][0],x_embed_cl_23[i][1],'.',color='red',markersize=6)
    if logit_23[i,1] == 1:
        axs[0,0].plot(x_embed_cl_23[i][0],x_embed_cl_23[i][1],'.',color='blue',markersize=10)
    axs[0, 0].set_title('A')

for i in range(length_23):
    if logit_23[i,0] == 1:
        axs[0,1].plot(x_embed_ce_23[i][0],x_embed_ce_23[i][1],'.',color='red',markersize=6)
    if logit_23[i,1] == 1:
        axs[0,1].plot(x_embed_ce_23[i][0],x_embed_ce_23[i][1],'.',color='blue',markersize=10)
    axs[0, 1].set_title('B')

for i in range(length_7):
    if logit_7[i,0] == 1:
        axs[1,0].plot(x_embed_cl_7[i][0],x_embed_cl_7[i][1],'.',color='red',markersize=6)
    if logit_7[i,1] == 1:
        axs[1,0].plot(x_embed_cl_7[i][0],x_embed_cl_7[i][1],'.',color='blue',markersize=10)
    axs[1, 0].set_title('C')

for i in range(length_7):
    if logit_7[i,0] == 1:
        axs[1,1].plot(x_embed_ce_7[i][0],x_embed_ce_7[i][1],'.',color='red',markersize=6)
    if logit_7[i,1] == 1:
        axs[1,1].plot(x_embed_ce_7[i][0],x_embed_ce_7[i][1],'.',color='blue',markersize=10)
    axs[1, 1].set_title('D')

plt.show()

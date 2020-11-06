from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

"""
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

#fig, axs = plt.subplots(6,2)

#fig.suptitle('Mortality Prediction')

for i in range(length_23):
    if logit_23[i,0] == 1:
        plt.plot(x_embed_cl_23[i][0],x_embed_cl_23[i][1],'.',color='red',markersize=6)
    if logit_23[i,1] == 1:
        plt.plot(x_embed_cl_23[i][0],x_embed_cl_23[i][1],'.',color='blue',markersize=10)
    #axs[0, 0].set_title('A')

plt.show()

for i in range(length_23):
    if logit_23[i,0] == 1:
        plt.plot(x_embed_ce_23[i][0],x_embed_ce_23[i][1],'.',color='red',markersize=6)
    if logit_23[i,1] == 1:
        plt.plot(x_embed_ce_23[i][0],x_embed_ce_23[i][1],'.',color='blue',markersize=10)
    #axs[0, 1].set_title('B')

plt.show()

for i in range(length_7):
    if logit_7[i,0] == 1:
        plt.plot(x_embed_cl_7[i][0],x_embed_cl_7[i][1],'.',color='red',markersize=6)
    if logit_7[i,1] == 1:
        plt.plot(x_embed_cl_7[i][0],x_embed_cl_7[i][1],'.',color='blue',markersize=10)
    #axs[1, 1].set_title('D')

plt.show()

for i in range(length_7):
    if logit_7[i,0] == 1:
        plt.plot(x_embed_ce_7[i][0],x_embed_ce_7[i][1],'.',color='red',markersize=6)
    if logit_7[i,1] == 1:
        plt.plot(x_embed_ce_7[i][0],x_embed_ce_7[i][1],'.',color='blue',markersize=10)
    #axs[1, 0].set_title('C')

plt.show()
#axs[0,0].text(6.1, 1.36, 'Mortality Prediction', color='r',fontsize=20)
#plt.show()
"""
"""
Vis for intubation
"""
"""
intubate_ce_5 = np.load('5_data_intubate_ce_24h.npy')
intubate_cl_5 = np.load('5_data_intubate_cl_24h.npy')
intubate_ce_10 = np.load('10_data_intubate_ce_24h.npy')
intubate_cl_10 = np.load('10_data_intubate_cl_24h.npy')
logit_5 = np.load('logit_5_intubate.npy')
logit_10 = np.load('logit_10_intubate.npy')

length_10 = intubate_ce_10.shape[0]
length_5 = intubate_ce_5.shape[0]
x_embed_ce_5 = TSNE(n_components=2).fit_transform(intubate_ce_5)
x_embed_cl_5 = TSNE(n_components=2).fit_transform(intubate_cl_5)
x_embed_ce_10 = TSNE(n_components=2).fit_transform(intubate_ce_10)
x_embed_cl_10 = TSNE(n_components=2).fit_transform(intubate_cl_10)

#fig, axs = plt.subplots(2,2)
#fig.suptitle('Intubation Prediction')

for i in range(length_10):
    if logit_10[i,0] == 1:
        plt.plot(x_embed_cl_10[i][0],x_embed_cl_10[i][1],'.',color='red',markersize=6)
    if logit_10[i,1] == 1:
        plt.plot(x_embed_cl_10[i][0],x_embed_cl_10[i][1],'.',color='blue',markersize=10)
    #plt.set_title('A')

plt.show()

for i in range(length_10):
    if logit_10[i,0] == 1:
        plt.plot(x_embed_ce_10[i][0],x_embed_ce_10[i][1],'.',color='red',markersize=6)
    if logit_10[i,1] == 1:
        plt.plot(x_embed_ce_10[i][0],x_embed_ce_10[i][1],'.',color='blue',markersize=10)
    #plt.set_title('B')

plt.show()

for i in range(length_5):
    if logit_5[i,0] == 1:
        plt.plot(x_embed_cl_5[i][0],x_embed_cl_5[i][1],'.',color='red',markersize=6)
    if logit_5[i,1] == 1:
        plt.plot(x_embed_cl_5[i][0],x_embed_cl_5[i][1],'.',color='blue',markersize=10)
    #axs[3, 0].set_title('C')

plt.show()

for i in range(length_5):
    if logit_5[i,0] == 1:
        plt.plot(x_embed_ce_5[i][0],x_embed_ce_5[i][1],'.',color='red',markersize=6)
    if logit_5[i,1] == 1:
        plt.plot(x_embed_ce_5[i][0],x_embed_ce_5[i][1],'.',color='blue',markersize=10)
    #axs[3, 1].set_title('D')

plt.show()
"""
#axs[2,0].text(6.1, 1.36, 'Intubation Prediction', color='b',fontsize=20)
#plt.show()


"""
Vis for ICU
"""

intubate_ce_5 = np.load('7_data_icu_ce_24h.npy')
intubate_cl_5 = np.load('7_data_icu_cl_24h.npy')
intubate_ce_10 = np.load('17_data_icu_ce_24h.npy')
intubate_cl_10 = np.load('17_data_icu_cl_24h.npy')
logit_5 = np.load('logit_7_icu.npy')
logit_10 = np.load('logit_17_icu.npy')

length_10 = intubate_ce_10.shape[0]
length_5 = intubate_ce_5.shape[0]
x_embed_ce_5 = TSNE(n_components=2).fit_transform(intubate_ce_5)
x_embed_cl_5 = TSNE(n_components=2).fit_transform(intubate_cl_5)
x_embed_ce_10 = TSNE(n_components=2).fit_transform(intubate_ce_10)
x_embed_cl_10 = TSNE(n_components=2).fit_transform(intubate_cl_10)


#fig, axs = plt.subplots(2,2)
#fig.suptitle('Icu Prediction')

for i in range(length_10):
    if logit_10[i,0] == 1:
        plt.plot(x_embed_cl_10[i][0],x_embed_cl_10[i][1],'.',color='red',markersize=6)
    if logit_10[i,1] == 1:
        plt.plot(x_embed_cl_10[i][0],x_embed_cl_10[i][1],'.',color='blue',markersize=10)
    #axs[4, 0].set_title('A')

plt.show()


for i in range(length_10):
    if logit_10[i,0] == 1:
        plt.plot(x_embed_ce_10[i][0],x_embed_ce_10[i][1],'.',color='red',markersize=6)
    if logit_10[i,1] == 1:
        plt.plot(x_embed_ce_10[i][0],x_embed_ce_10[i][1],'.',color='blue',markersize=10)
    #axs[4, 1].set_title('B')
plt.show()

for i in range(length_5):
    if logit_5[i,0] == 1:
        plt.plot(x_embed_cl_5[i][0],x_embed_cl_5[i][1],'.',color='red',markersize=6)
    if logit_5[i,1] == 1:
        plt.plot(x_embed_cl_5[i][0],x_embed_cl_5[i][1],'.',color='blue',markersize=10)
    #plt.set_title('C')

plt.show()

for i in range(length_5):
    if logit_5[i,0] == 1:
        plt.plot(x_embed_ce_5[i][0],x_embed_ce_5[i][1],'.',color='red',markersize=6)
    if logit_5[i,1] == 1:
        plt.plot(x_embed_ce_5[i][0],x_embed_ce_5[i][1],'.',color='blue',markersize=10)
    #axs[5, 1].set_title('D')

#axs[4,0].text(6.1, 1.36, 'ICU Transfer Precition', color='b',fontsize=20)
plt.show()

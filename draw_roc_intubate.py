import numpy as np
import matplotlib.pyplot as plt



tp_rates =

fp_rates =

tp_rates_CL =

fp_rates_cl =

tp_rate_ce_retain=

fp_rates_ce_retain=

tp_rates_CL_RNN =

fp_rates_cl_RNN =

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Intubation Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(fp_rates, tp_rates, color='green', linewidth=1, linestyle='dashed',label='RNN+CE(AUC=0.818)')


plt.plot(fp_rates_ce_retain,tp_rate_ce_retain,color='blue',linestyle='dashed',label='RETAIN+CE(AUC=0.784)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(fp_rates_cl_RNN,tp_rates_CL_RNN,color='violet',linewidth=1.5,label='RNN+CL(AUC=0.887)')

plt.plot(fp_rates_cl, tp_rates_CL, color='red', linewidth=1.5, label='RETAIN+CL(AUC=0.878)')



plt.legend(loc='lower right')
plt.show()
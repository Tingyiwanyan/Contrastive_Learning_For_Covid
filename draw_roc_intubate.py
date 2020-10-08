import numpy as np
import matplotlib.pyplot as plt


tp_rates = [1.0,
 1.0,
 1.0,
 0.9854337899543378,
 0.9708675799086758,
 0.9663013698630136,
 0.9526027397260274,
 0.944337899543379,
 0.9269406392694064,
 0.8767123287671232,
 0.8493150684931506,
 0.8310502283105022,
 0.7945205479452054,
 0.771689497716895,
 0.7397260273972602,
 0.7351598173515982,
 0.726027397260274,
 0.7123287671232876,
 0.7031963470319634,
 0.684931506849315,
 0.6757990867579908,
 0.6712328767123288,
 0.6621004566210046,
 0.6529680365296804,
 0.6529680365296804,
 0.639269406392694,
 0.6210045662100456,
 0.6210045662100456,
 0.6210045662100456,
 0.6118721461187214,
 0.6027397260273972,
 0.5981735159817352,
 0.593607305936073,
 0.589041095890411,
 0.589041095890411,
 0.589041095890411,
 0.589041095890411,
 0.5799086757990868,
 0.5753424657534246,
 0.5753424657534246,
 0.5707762557077626,
 0.5616438356164384,
 0.5616438356164384,
 0.5525114155251142,
 0.547945205479452,
 0.54337899543379,
 0.54337899543379,
 0.5388127853881278,
 0.5342465753424658,
 0.5296803652968036,
 0.5296803652968036,
 0.5296803652968036,
 0.5296803652968036,
 0.5251141552511416,
 0.5205479452054794,
 0.5159817351598174,
 0.5114155251141552,
 0.5114155251141552,
 0.502283105022831,
 0.4885844748858447,
 0.4885844748858447,
 0.4840182648401826,
 0.4794520547945205,
 0.4794520547945205,
 0.4794520547945205,
 0.4703196347031963,
 0.4703196347031963,
 0.4657534246575342,
 0.4657534246575342,
 0.4611872146118721,
 0.4611872146118721,
 0.45662100456621,
 0.4520547945205479,
 0.4474885844748858,
 0.4474885844748858,
 0.4429223744292237,
 0.4429223744292237,
 0.4383561643835616,
 0.4337899543378995,
 0.4292237442922374,
 0.4246575342465753,
 0.4246575342465753,
 0.4200913242009132,
 0.4155251141552511,
 0.410958904109589,
 0.4063926940639269,
 0.4018264840182648,
 0.3926940639269406,
 0.3926940639269406,
 0.3835616438356164,
 0.3698630136986301,
 0.3698630136986301,
 0.3561643835616438,
 0.3515981735159817,
 0.3424657534246575,
 0.3333333333333333,
 0.3242009132420091,
 0.3013698630136986,
 0.2831050228310502,
 0.2465753424657534,
 0.0]

fp_rates = [1.0,
 0.9625390218522373,
 0.8678459937565036,
 0.7991675338189386,
 0.7117585848074922,
 0.617585848074922,
 0.5171696149843913,
 0.42351716961498437,
 0.33350676378772115,
 0.2643080124869927,
 0.19406867845993755,
 0.14516129032258066,
 0.09365244536940687,
 0.0691987513007284,
 0.05463059313215401,
 0.04630593132154006,
 0.04266389177939646,
 0.040062434963579606,
 0.03485952133194589,
 0.03433922996878252,
 0.033818938605619145,
 0.029136316337148804,
 0.027055150884495317,
 0.023413111342351717,
 0.022892819979188347,
 0.02081165452653486,
 0.019771071800208116,
 0.019771071800208116,
 0.019250780437044746,
 0.019250780437044746,
 0.019250780437044746,
 0.01716961498439126,
 0.01716961498439126,
 0.01716961498439126,
 0.015608740894901144,
 0.014568158168574402,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.013527575442247659,
 0.013527575442247659,
 0.013007284079084287,
 0.013007284079084287,
 0.011966701352757543,
 0.011446409989594173,
 0.010926118626430802,
 0.010926118626430802,
 0.010926118626430802,
 0.010926118626430802,
 0.009885535900104058,
 0.009885535900104058,
 0.009365244536940686,
 0.008844953173777315,
 0.008844953173777315,
 0.008844953173777315,
 0.008844953173777315,
 0.008844953173777315,
 0.007284079084287201,
 0.007284079084287201,
 0.007284079084287201,
 0.007284079084287201,
 0.007284079084287201,
 0.007284079084287201,
 0.006763787721123829,
 0.006243496357960458,
 0.005723204994797087,
 0.005723204994797087,
 0.005202913631633715,
 0.004682622268470343,
 0.004682622268470343,
 0.004682622268470343,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.003121748178980229,
 0.0026014568158168575,
 0.0026014568158168575,
 0.0026014568158168575,
 0.002081165452653486,
 0.002081165452653486,
 0.002081165452653486,
 0.0015608740894901144,
 0.0]

tp_rates_CL = [1.0,
 0.9269406392694064,
 0.8584474885844748,
 0.8493150684931506,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8264840182648402,
 0.8264840182648402,
 0.8264840182648402,
 0.821917808219178,
 0.821917808219178,
 0.817351598173516,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8082191780821918,
 0.8036529680365296,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7945205479452054,
 0.7899543378995434,
 0.7853881278538812,
 0.7853881278538812,
 0.7853881278538812,
 0.776255707762557,
 0.771689497716895,
 0.7579908675799086,
 0.730593607305936,
 0.5753424657534246,
 0.0]

fp_rates_cl = [1.0,
 0.25286160249739853,
 0.047866805411030174,
 0.03902185223725286,
 0.03590010405827263,
 0.031217481789802288,
 0.030697190426638918,
 0.029656607700312174,
 0.029656607700312174,
 0.029136316337148804,
 0.02861602497398543,
 0.02809573361082206,
 0.027055150884495317,
 0.027055150884495317,
 0.026534859521331947,
 0.026534859521331947,
 0.026534859521331947,
 0.026014568158168574,
 0.025494276795005204,
 0.025494276795005204,
 0.025494276795005204,
 0.025494276795005204,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02497398543184183,
 0.02445369406867846,
 0.023413111342351717,
 0.023413111342351717,
 0.023413111342351717,
 0.023413111342351717,
 0.022892819979188347,
 0.022892819979188347,
 0.02081165452653486,
 0.018730489073881373,
 0.013007284079084287,
 0.0]

tp_rate_ce_retain=[1.0,
 0.9908675799086758,
 0.9771689497716894,
 0.954337899543379,
 0.9178082191780822,
 0.8584474885844748,
 0.8264840182648402,
 0.7990867579908676,
 0.7808219178082192,
 0.776255707762557,
 0.7625570776255708,
 0.7625570776255708,
 0.7397260273972602,
 0.730593607305936,
 0.7168949771689498,
 0.7168949771689498,
 0.7168949771689498,
 0.6986301369863014,
 0.6986301369863014,
 0.6940639269406392,
 0.6894977168949772,
 0.680365296803653,
 0.6666666666666666,
 0.6666666666666666,
 0.6621004566210046,
 0.6575342465753424,
 0.6529680365296804,
 0.634703196347032,
 0.6255707762557078,
 0.6164383561643836,
 0.6118721461187214,
 0.6118721461187214,
 0.5981735159817352,
 0.593607305936073,
 0.5799086757990868,
 0.5753424657534246,
 0.5707762557077626,
 0.5616438356164384,
 0.5616438356164384,
 0.5570776255707762,
 0.54337899543379,
 0.5342465753424658,
 0.5205479452054794,
 0.5205479452054794,
 0.5114155251141552,
 0.5068493150684932,
 0.4977168949771689,
 0.4977168949771689,
 0.4885844748858447,
 0.4840182648401826,
 0.4794520547945205,
 0.4794520547945205,
 0.4794520547945205,
 0.4748858447488584,
 0.4657534246575342,
 0.4611872146118721,
 0.45662100456621,
 0.4474885844748858,
 0.4474885844748858,
 0.4383561643835616,
 0.4246575342465753,
 0.4246575342465753,
 0.410958904109589,
 0.410958904109589,
 0.4063926940639269,
 0.4063926940639269,
 0.3926940639269406,
 0.3926940639269406,
 0.3881278538812785,
 0.3881278538812785,
 0.3789954337899543,
 0.3744292237442922,
 0.3698630136986301,
 0.3607305936073059,
 0.3607305936073059,
 0.3424657534246575,
 0.3287671232876712,
 0.3242009132420091,
 0.3242009132420091,
 0.3059360730593607,
 0.3013698630136986,
 0.2876712328767123,
 0.2785388127853881,
 0.2648401826484018,
 0.2557077625570776,
 0.2557077625570776,
 0.2374429223744292,
 0.2191780821917808,
 0.2191780821917808,
 0.2146118721461187,
 0.2100456621004566,
 0.1963470319634703,
 0.1963470319634703,
 0.182648401826484,
 0.1506849315068493,
 0.1415525114155251,
 0.1278538812785388,
 0.1095890410958904,
 0.091324200913242,
 0.0821917808219178,
 0.0]

fp_rates_ce_retain=[1.0,
 0.95369406867846,
 0.8329864724245577,
 0.7336108220603538,
 0.595213319458897,
 0.3392299687825182,
 0.16441207075962538,
 0.12382934443288242,
 0.06971904266389178,
 0.06087408949011446,
 0.05359001040582726,
 0.04890738813735692,
 0.04630593132154006,
 0.04162330905306972,
 0.041103017689906346,
 0.03954214360041623,
 0.03850156087408949,
 0.036420395421436005,
 0.036420395421436005,
 0.032778355879292405,
 0.030697190426638918,
 0.029136316337148804,
 0.02861602497398543,
 0.027055150884495317,
 0.026014568158168574,
 0.023933402705515087,
 0.022892819979188347,
 0.022372528616024973,
 0.022372528616024973,
 0.02133194588969823,
 0.02029136316337149,
 0.019250780437044746,
 0.01768990634755463,
 0.01716961498439126,
 0.01716961498439126,
 0.01716961498439126,
 0.01716961498439126,
 0.015608740894901144,
 0.015608740894901144,
 0.015088449531737774,
 0.015088449531737774,
 0.014568158168574402,
 0.012486992715920915,
 0.010926118626430802,
 0.010926118626430802,
 0.010926118626430802,
 0.009885535900104058,
 0.009885535900104058,
 0.009365244536940686,
 0.009365244536940686,
 0.008844953173777315,
 0.008324661810613945,
 0.007804370447450572,
 0.007284079084287201,
 0.006763787721123829,
 0.006763787721123829,
 0.006243496357960458,
 0.006243496357960458,
 0.006243496357960458,
 0.006243496357960458,
 0.005723204994797087,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.005202913631633715,
 0.004682622268470343,
 0.004682622268470343,
 0.004682622268470343,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.004162330905306972,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.0036420395421436005,
 0.003121748178980229,
 0.0026014568158168575,
 0.002081165452653486,
 0.001040582726326743,
 0.001040582726326743,
 0.001040582726326743,
 0.001040582726326743,
 0.001040582726326743,
 0.001040582726326743,
 0.0005202913631633715,
 0.0005202913631633715,
 0.0]

tp_rate_hl_retain=[1.0,
 0.7579908675799086,
 0.730593607305936,
 0.7168949771689498,
 0.7077625570776256,
 0.6894977168949772,
 0.6894977168949772,
 0.680365296803653,
 0.6757990867579908,
 0.6712328767123288,
 0.6712328767123288,
 0.6712328767123288,
 0.6666666666666666,
 0.6666666666666666,
 0.6621004566210046,
 0.6621004566210046,
 0.6621004566210046,
 0.6621004566210046,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6575342465753424,
 0.6484018264840182,
 0.6484018264840182,
 0.6484018264840182,
 0.6484018264840182,
 0.6484018264840182,
 0.6484018264840182,
 0.6438356164383562,
 0.6438356164383562,
 0.639269406392694,
 0.639269406392694,
 0.634703196347032,
 0.634703196347032,
 0.634703196347032,
 0.634703196347032,
 0.6301369863013698,
 0.6301369863013698,
 0.6301369863013698,
 0.6210045662100456,
 0.6027397260273972,
 0.5981735159817352,
 0.589041095890411,
 0.589041095890411,
 0.5844748858447488,
 0.5844748858447488,
 0.5844748858447488,
 0.5844748858447488,
 0.5799086757990868,
 0.5707762557077626,
 0.5662100456621004,
 0.5662100456621004,
 0.5662100456621004,
 0.5616438356164384,
 0.5616438356164384,
 0.5570776255707762,
 0.5570776255707762,
 0.5570776255707762,
 0.5570776255707762,
 0.5570776255707762,
 0.5570776255707762,
 0.5525114155251142,
 0.547945205479452,
 0.547945205479452,
 0.54337899543379,
 0.5388127853881278,
 0.5388127853881278,
 0.5388127853881278,
 0.5388127853881278,
 0.5342465753424658,
 0.5342465753424658,
 0.5342465753424658,
 0.5342465753424658,
 0.5342465753424658,
 0.5342465753424658,
 0.5342465753424658,
 0.5296803652968036,
 0.5296803652968036,
 0.5296803652968036,
 0.5296803652968036,
 0.5251141552511416,
 0.5159817351598174,
 0.5114155251141552,
 0.5114155251141552,
 0.502283105022831,
 0.502283105022831,
 0.4885844748858447,
 0.4840182648401826,
 0.4794520547945205,
 0.4657534246575342,
 0.4520547945205479,
 0.4429223744292237,
 0.4200913242009132,
 0.3972602739726027,
 0.0]

fp_rate_hl_retain=[0.9994797086368367,
 0.05827263267429761,
 0.049427679500520294,
 0.04266389177939646,
 0.04058272632674298,
 0.03798126951092612,
 0.03798126951092612,
 0.03694068678459938,
 0.03485952133194589,
 0.033818938605619145,
 0.03329864724245578,
 0.03225806451612903,
 0.030176899063475548,
 0.030176899063475548,
 0.029656607700312174,
 0.029136316337148804,
 0.029136316337148804,
 0.029136316337148804,
 0.029136316337148804,
 0.02861602497398543,
 0.02809573361082206,
 0.027055150884495317,
 0.026534859521331947,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.025494276795005204,
 0.025494276795005204,
 0.02445369406867846,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023413111342351717,
 0.022892819979188347,
 0.022892819979188347,
 0.022372528616024973,
 0.021852237252861603,
 0.021852237252861603,
 0.02133194588969823,
 0.02081165452653486,
 0.02081165452653486,
 0.02029136316337149,
 0.02029136316337149,
 0.02029136316337149,
 0.019771071800208116,
 0.019250780437044746,
 0.019250780437044746,
 0.018210197710718003,
 0.018210197710718003,
 0.018210197710718003,
 0.01768990634755463,
 0.01768990634755463,
 0.01716961498439126,
 0.01716961498439126,
 0.01664932362122789,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.015608740894901144,
 0.015608740894901144,
 0.015088449531737774,
 0.015088449531737774,
 0.014568158168574402,
 0.014568158168574402,
 0.014568158168574402,
 0.014568158168574402,
 0.014568158168574402,
 0.014568158168574402,
 0.01404786680541103,
 0.013527575442247659,
 0.013007284079084287,
 0.011966701352757543,
 0.011966701352757543,
 0.011446409989594173,
 0.011446409989594173,
 0.009885535900104058,
 0.009885535900104058,
 0.009885535900104058,
 0.009885535900104058,
 0.009885535900104058,
 0.009365244536940686,
 0.009365244536940686,
 0.008844953173777315,
 0.007804370447450572,
 0.007284079084287201,
 0.007284079084287201,
 0.007284079084287201,
 0.006763787721123829,
 0.0]

tp_rates_CL_RNN = [1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 0.9178082191780822,
 0.8904109589041096,
 0.8812785388127854,
 0.8812785388127854,
 0.8767123287671232,
 0.8767123287671232,
 0.8767123287671232,
 0.867579908675799,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.863013698630137,
 0.8584474885844748,
 0.8584474885844748,
 0.8584474885844748,
 0.8584474885844748,
 0.8584474885844748,
 0.8584474885844748,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8538812785388128,
 0.8493150684931506,
 0.8493150684931506,
 0.8493150684931506,
 0.8493150684931506,
 0.8493150684931506,
 0.8493150684931506,
 0.8447488584474886,
 0.8401826484018264,
 0.8401826484018264,
 0.8401826484018264,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8356164383561644,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8310502283105022,
 0.8264840182648402,
 0.8264840182648402,
 0.8264840182648402,
 0.8264840182648402,
 0.821917808219178,
 0.821917808219178,
 0.821917808219178,
 0.821917808219178,
 0.821917808219178,
 0.821917808219178,
 0.817351598173516,
 0.817351598173516,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8127853881278538,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8082191780821918,
 0.8036529680365296,
 0.8036529680365296,
 0.8036529680365296,
 0.8036529680365296,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7990867579908676,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7945205479452054,
 0.7899543378995434,
 0.7808219178082192,
 0.776255707762557,
 0.771689497716895,
 0.771689497716895,
 0.7488584474885844,
 0.7488584474885844,
 0.7442922374429224,
 0.7442922374429224,
 0.7351598173515982,
 0.730593607305936,
 0.7214611872146118,
 0.6986301369863014,
 0.6757990867579908,
 0.6575342465753424,
 0.593607305936073,
 0.5251141552511416,
 0.4246575342465753,
 0.228310502283105,
 0.0365296803652968,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]

fp_rates_cl_RNN = [1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 0.9947970863683663,
 0.186264308012487,
 0.11082206035379813,
 0.07596253902185224,
 0.06191467221644121,
 0.054110301768990635,
 0.04734651404786681,
 0.04422476586888657,
 0.04058272632674298,
 0.040062434963579606,
 0.03902185223725286,
 0.037460978147762745,
 0.03590010405827263,
 0.03485952133194589,
 0.03433922996878252,
 0.03225806451612903,
 0.030176899063475548,
 0.02861602497398543,
 0.02861602497398543,
 0.027575442247658687,
 0.026534859521331947,
 0.026534859521331947,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.026014568158168574,
 0.025494276795005204,
 0.02445369406867846,
 0.023933402705515087,
 0.023933402705515087,
 0.023933402705515087,
 0.023413111342351717,
 0.022892819979188347,
 0.022892819979188347,
 0.022892819979188347,
 0.022892819979188347,
 0.022892819979188347,
 0.022892819979188347,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.022372528616024973,
 0.021852237252861603,
 0.021852237252861603,
 0.021852237252861603,
 0.02133194588969823,
 0.02133194588969823,
 0.02133194588969823,
 0.02133194588969823,
 0.02081165452653486,
 0.02029136316337149,
 0.02029136316337149,
 0.02029136316337149,
 0.02029136316337149,
 0.019771071800208116,
 0.019771071800208116,
 0.019771071800208116,
 0.019771071800208116,
 0.019771071800208116,
 0.019250780437044746,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018730489073881373,
 0.018210197710718003,
 0.018210197710718003,
 0.018210197710718003,
 0.018210197710718003,
 0.01768990634755463,
 0.01768990634755463,
 0.01716961498439126,
 0.01716961498439126,
 0.01716961498439126,
 0.01664932362122789,
 0.01664932362122789,
 0.01664932362122789,
 0.01664932362122789,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.016129032258064516,
 0.015608740894901144,
 0.015608740894901144,
 0.015608740894901144,
 0.015608740894901144,
 0.015608740894901144,
 0.015088449531737774,
 0.015088449531737774,
 0.015088449531737774,
 0.015088449531737774,
 0.015088449531737774,
 0.015088449531737774,
 0.014568158168574402,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.01404786680541103,
 0.013527575442247659,
 0.013527575442247659,
 0.013527575442247659,
 0.013527575442247659,
 0.013527575442247659,
 0.013527575442247659,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.013007284079084287,
 0.012486992715920915,
 0.012486992715920915,
 0.012486992715920915,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.011966701352757543,
 0.010926118626430802,
 0.010926118626430802,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.01040582726326743,
 0.009365244536940686,
 0.008844953173777315,
 0.008324661810613945,
 0.008324661810613945,
 0.006763787721123829,
 0.003121748178980229,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Death Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(fp_rates, tp_rates, color='green', linewidth=1, label='RNN+CE')


plt.plot(fp_rates_ce_retain,tp_rate_ce_retain,color='blue',label='RETAIN+CE')

plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(fp_rates_cl, tp_rates_CL, color='red', linewidth=1, label='RETAIN+CL')

plt.plot(fp_rates_cl_RNN,tp_rates_CL_RNN,color='violet',label='RNN+CL')

plt.legend(loc='lower right')
plt.show()
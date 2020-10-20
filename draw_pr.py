import numpy as np
import matplotlib.pyplot as plt

precision_ce_death = []


recall_ce_death = []

precition_cl_retain = [0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19803370786516855,
 0.19797525309336333,
 0.20880428316478286,
 0.2859013531209079,
 0.33507853403141363,
 0.38161898965307367,
 0.4232081911262799,
 0.4599850411368736,
 0.49033816425120774,
 0.5189655172413793,
 0.5404178019981835,
 0.55765595463138,
 0.5671497584541063,
 0.5765407554671969,
 0.5845997973657548,
 0.5981308411214953,
 0.6061246040126715,
 0.6121794871794872,
 0.6157158234660925,
 0.6191512513601741,
 0.6212952799121844,
 0.6286353467561522,
 0.6375,
 0.6417910447761194,
 0.6487747957992999,
 0.6525323910482921,
 0.6583333333333333,
 0.6598802395209581,
 0.6654589371980676,
 0.670316301703163,
 0.6719512195121952,
 0.6735941320293398,
 0.6781750924784217,
 0.6786600496277916,
 0.6791510611735331,
 0.6808510638297872,
 0.6813048933500627,
 0.683944374209861,
 0.6861499364675985,
 0.686624203821656,
 0.6883780332056194,
 0.6932989690721649,
 0.6987012987012987,
 0.6987012987012987,
 0.7023498694516971,
 0.7026315789473684,
 0.7054161162483488,
 0.7063492063492064,
 0.7068965517241379,
 0.7083888149134487,
 0.7102803738317757,
 0.7121820615796519,
 0.7150537634408602,
 0.7150537634408602,
 0.7175675675675676,
 0.7191316146540027,
 0.7191316146540027,
 0.720708446866485,
 0.7219178082191781,
 0.721763085399449,
 0.7233748271092669,
 0.724376731301939,
 0.7266387726638772,
 0.7276536312849162,
 0.7276536312849162,
 0.7286713286713287,
 0.7303370786516854,
 0.7320169252468265,
 0.7326732673267327,
 0.7326732673267327,
 0.734375,
 0.7375178316690443,
 0.7374461979913917,
 0.7398843930635838,
 0.7409551374819102,
 0.7438136826783115,
 0.7445255474452555,
 0.7456140350877193,
 0.7452415812591509,
 0.7463343108504399,
 0.7474302496328928,
 0.7496318114874816,
 0.7507374631268436,
 0.7518463810930576,
 0.7529585798816568,
 0.7540740740740741,
 0.7540740740740741,
 0.7555886736214605,
 0.7567164179104477,
 0.7567164179104477,
 0.7563527653213752,
 0.7593984962406015,
 0.7616892911010558,
 0.7606060606060606,
 0.7617602427921093,
 0.7637195121951219,
 0.7648854961832061,
 0.7695852534562212,
 0.7707692307692308,
 0.7727975270479135,
 0.7739938080495357,
 0.7741433021806854,
 0.7741433021806854,
 0.7737909516380655,
 0.7755102040816326,
 0.7751572327044025,
 0.777602523659306,
 0.7797147385103012,
 0.7790143084260731,
 0.7795527156549521,
 0.7792,
 0.7813504823151125,
 0.7813504823151125,
 0.7838709677419354,
 0.7844408427876823,
 0.7853658536585366,
 0.7862969004893964,
 0.7875816993464052,
 0.7872340425531915,
 0.787828947368421,
 0.7904290429042904,
 0.7900826446280992,
 0.7893864013266998,
 0.7929883138564274,
 0.7919463087248322,
 0.7946127946127947,
 0.7956081081081081,
 0.7949152542372881,
 0.7959183673469388,
 0.7959183673469388,
 0.7982905982905983,
 0.803448275862069,
 0.8044982698961938,
 0.8072916666666666,
 0.8097731239092496,
 0.8097731239092496,
 0.8098591549295775,
 0.8117229129662522,
 0.8113879003558719,
 0.8128342245989305,
 0.8154121863799283,
 0.8176895306859205,
 0.8218181818181818,
 0.8244972577696527,
 0.8280961182994455,
 0.8280961182994455,
 0.8314606741573034,
 0.8301886792452831,
 0.8342857142857143,
 0.8403846153846154,
 0.8434442270058709,
 0.846,
 0.847870182555781,
 0.8503118503118503,
 0.8565400843881856,
 0.8571428571428571,
 0.8586956521739131,
 0.8628318584070797,
 0.8687782805429864,
 0.8767123287671232,
 0.8764568764568764,
 0.8803827751196173,
 0.8810679611650486,
 0.8936708860759494,
 0.9038961038961039,
 0.9110512129380054,
 0.9146005509641874,
 0.9195402298850575,
 0.9256965944272446,
 0.9311475409836065,
 0.9359430604982206,
 0.9352226720647774,
 0.9523809523809523,
 0.9590643274853801,
 0.9761904761904762,
 0.9888888888888889,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0]

recall_cl_retain_death = [1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 0.9985815602836879,
 0.9957446808510638,
 0.9290780141843972,
 0.9078014184397163,
 0.8893617021276595,
 0.8794326241134752,
 0.8723404255319149,
 0.8638297872340426,
 0.8539007092198582,
 0.8439716312056738,
 0.8368794326241135,
 0.8326241134751773,
 0.8226950354609929,
 0.8184397163120567,
 0.8170212765957446,
 0.8141843971631205,
 0.8127659574468085,
 0.8113475177304964,
 0.8070921985815603,
 0.8028368794326242,
 0.7971631205673759,
 0.7957446808510639,
 0.7929078014184398,
 0.7886524822695036,
 0.7858156028368795,
 0.7843971631205674,
 0.7815602836879433,
 0.7815602836879433,
 0.7815602836879433,
 0.7815602836879433,
 0.7815602836879433,
 0.7801418439716312,
 0.775886524822695,
 0.7716312056737589,
 0.7716312056737589,
 0.7702127659574468,
 0.7673758865248227,
 0.7659574468085106,
 0.7645390070921986,
 0.7645390070921986,
 0.7631205673758865,
 0.7631205673758865,
 0.7631205673758865,
 0.7631205673758865,
 0.7574468085106383,
 0.7574468085106383,
 0.7574468085106383,
 0.7560283687943262,
 0.7546099290780142,
 0.7546099290780142,
 0.7546099290780142,
 0.7546099290780142,
 0.7546099290780142,
 0.7531914893617021,
 0.75177304964539,
 0.75177304964539,
 0.750354609929078,
 0.7475177304964539,
 0.7432624113475177,
 0.7418439716312056,
 0.7418439716312056,
 0.7390070921985815,
 0.7390070921985815,
 0.7390070921985815,
 0.7390070921985815,
 0.7375886524822695,
 0.7361702127659574,
 0.7347517730496453,
 0.7347517730496453,
 0.7333333333333333,
 0.7333333333333333,
 0.7290780141843972,
 0.7262411347517731,
 0.7262411347517731,
 0.724822695035461,
 0.723404255319149,
 0.723404255319149,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7219858156028369,
 0.7191489361702128,
 0.7191489361702128,
 0.7191489361702128,
 0.7177304964539007,
 0.7163120567375887,
 0.7163120567375887,
 0.7120567375886525,
 0.7120567375886525,
 0.7106382978723405,
 0.7106382978723405,
 0.7106382978723405,
 0.7106382978723405,
 0.7092198581560284,
 0.7092198581560284,
 0.7049645390070922,
 0.7049645390070922,
 0.7035460992907802,
 0.700709219858156,
 0.699290780141844,
 0.699290780141844,
 0.6978723404255319,
 0.6950354609929078,
 0.6921985815602837,
 0.6907801418439716,
 0.6893617021276596,
 0.6893617021276596,
 0.6893617021276596,
 0.6865248226950355,
 0.6851063829787234,
 0.6836879432624113,
 0.6836879432624113,
 0.6822695035460993,
 0.6794326241134752,
 0.6794326241134752,
 0.6780141843971631,
 0.675177304964539,
 0.6737588652482269,
 0.6695035460992907,
 0.6695035460992907,
 0.6680851063829787,
 0.6652482269503546,
 0.6638297872340425,
 0.6638297872340425,
 0.6624113475177305,
 0.6609929078014184,
 0.6595744680851063,
 0.6595744680851063,
 0.6581560283687943,
 0.6581560283687943,
 0.6524822695035462,
 0.64822695035461,
 0.6468085106382979,
 0.6468085106382979,
 0.6453900709219859,
 0.6425531914893617,
 0.6411347517730497,
 0.6397163120567376,
 0.6354609929078014,
 0.6354609929078014,
 0.6297872340425532,
 0.624113475177305,
 0.6212765957446809,
 0.6198581560283688,
 0.6113475177304964,
 0.6,
 0.5929078014184397,
 0.5801418439716312,
 0.5758865248226951,
 0.5702127659574469,
 0.5602836879432624,
 0.5531914893617021,
 0.5446808510638298,
 0.5446808510638298,
 0.5333333333333333,
 0.5219858156028369,
 0.5148936170212766,
 0.500709219858156,
 0.49361702127659574,
 0.4794326241134752,
 0.47092198581560285,
 0.45390070921985815,
 0.42411347517730497,
 0.40283687943262414,
 0.3730496453900709,
 0.3276595744680851,
 0.28368794326241137,
 0.2326241134751773,
 0.17446808510638298,
 0.12624113475177304,
 0.07659574468085106,
 0.02553191489361702,
 0.0070921985815602835,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]

precision_ce_rnn_death = []

recall_ce_rnn_death_24 = []

precision_cl_rnn_death=[]

recall_cl_rnn_death_24 = []




plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mortality Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(recall_ce_rnn_death_24, precision_ce_rnn_death, color='green', linestyle='dashed',linewidth=2, label='RNN+CE(AUC=0.875)')


plt.plot(recall_ce_death,precision_ce_death,color='blue',linestyle='dashed',linewidth=2,label='RETAIN+CE(AUC=0.888)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(precision_cl_rnn_death,recall_cl_rnn_death_24,color='violet',linewidth=1.5,label='RNN+CL(AUC=0.928)')
plt.plot(recall_cl_retain_death, precition_cl_retain, color='red', linewidth=1.5, label='RETAIN+CL(AUC=0.913)')


plt.legend(loc='lower right')
plt.show()
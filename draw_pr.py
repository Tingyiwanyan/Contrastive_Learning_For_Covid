import numpy as np
import matplotlib.pyplot as plt

precision_ce_death = [0.04586847, 0.04586847, 0.0461425 , 0.06041199, 0.07545766,
       0.09410013, 0.11661155, 0.14046628, 0.1666123 , 0.19063054,
       0.21460858, 0.2390453 , 0.26473868, 0.27871657, 0.30353278,
       0.31980136, 0.33404652, 0.35044419, 0.36021252, 0.37945369,
       0.39522872, 0.40387753, 0.41564071, 0.42590796, 0.43758593,
       0.45551012, 0.46707907, 0.47874868, 0.49063754, 0.50269142,
       0.51052113, 0.51828102, 0.52710264, 0.53475205, 0.54172214,
       0.54517609, 0.55539617, 0.55665526, 0.55620528, 0.55826857,
       0.56489337, 0.56687619, 0.57293048, 0.58333902, 0.58856331,
       0.59022123, 0.59628563, 0.5980247 , 0.60707099, 0.60874606,
       0.6125211 , 0.6155485 , 0.6169967 , 0.61946202, 0.62280142,
       0.62653612, 0.62959354, 0.63159238, 0.63375444, 0.63685155,
       0.63894623, 0.64571411, 0.64746802, 0.64967837, 0.6550835 ,
       0.65732855, 0.65933505, 0.66125296, 0.66284026, 0.66376672,
       0.66789858, 0.67142341, 0.67294872, 0.6755383 , 0.67816357,
       0.67816357, 0.67972133, 0.67933567, 0.67971311, 0.68547065,
       0.68967903, 0.68897845, 0.69117846, 0.69175466, 0.69418433,
       0.6983335 , 0.70027625, 0.70334871, 0.70518736, 0.70566544,
       0.71060829, 0.71123181, 0.71006477, 0.71070579, 0.71063245,
       0.71063245, 0.71129173, 0.71263873, 0.7185221 , 0.7217705 ,
       0.72153708, 0.72302626, 0.72376899, 0.72673552, 0.73153128,
       0.73243043, 0.7316555 , 0.73240712, 0.73240712, 0.73317038,
       0.73767287, 0.74348628, 0.74219267, 0.74224798, 0.74515494,
       0.7472077 , 0.74825445, 0.75049853, 0.75368035, 0.75345064,
       0.75522517, 0.75765438, 0.75618621, 0.75954051, 0.76042546,
       0.76134289, 0.76103217, 0.76123592, 0.76282279, 0.76552605,
       0.76479141, 0.76784225, 0.76719936, 0.76795435, 0.77551888,
       0.78536423, 0.78442458, 0.78843563, 0.79608117, 0.79428779,
       0.79292419, 0.79405661, 0.79237904, 0.79389348, 0.7937752 ,
       0.79582181, 0.79900418, 0.80186416, 0.80091814, 0.79999579,
       0.79870878, 0.80461928, 0.81235297, 0.82165435, 0.8261633 ,
       0.82423217, 0.83657972, 0.83473572, 0.82518192, 0.84085968,
       0.84283622, 0.83919494, 0.8475937 , 0.8631242 , 0.85877847,
       0.86394382, 0.88926185, 0.9057383 , 0.91098638, 0.91815156,
       0.91691835, 0.92068073, 0.93085213, 0.93409407, 0.93178244,
       0.9400303 , 0.93602399, 0.93274132, 0.92871259, 0.9452381 ,
       0.94095238, 0.93652174, 0.93126984, 0.9285213 , 0.91577965,
       0.92669683, 0.94      , 0.9       , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]


recall_ce_death = [1.        , 1.        , 1.        , 0.95294118, 0.93823529,
       0.90735294, 0.87058824, 0.86470588, 0.82647059, 0.79558824,
       0.74558824, 0.72058824, 0.70294118, 0.68823529, 0.67647059,
       0.67058824, 0.66176471, 0.65588235, 0.64558824, 0.61764706,
       0.60147059, 0.58382353, 0.57058824, 0.56029412, 0.55147059,
       0.54558824, 0.53970588, 0.53235294, 0.52941176, 0.52794118,
       0.52647059, 0.525     , 0.525     , 0.525     , 0.52058824,
       0.51617647, 0.51617647, 0.50882353, 0.50294118, 0.50147059,
       0.50147059, 0.50147059, 0.50147059, 0.50147059, 0.50147059,
       0.5       , 0.49852941, 0.49558824, 0.49411765, 0.49117647,
       0.48823529, 0.48823529, 0.48823529, 0.48676471, 0.48676471,
       0.48529412, 0.48235294, 0.48088235, 0.47941176, 0.47794118,
       0.47794118, 0.47647059, 0.47647059, 0.47647059, 0.47352941,
       0.47352941, 0.47205882, 0.47205882, 0.47205882, 0.47058824,
       0.46911765, 0.46764706, 0.46764706, 0.46617647, 0.46617647,
       0.46617647, 0.46470588, 0.46029412, 0.45882353, 0.45882353,
       0.45588235, 0.45441176, 0.45294118, 0.45294118, 0.45294118,
       0.45294118, 0.45147059, 0.45147059, 0.45      , 0.44852941,
       0.44852941, 0.44852941, 0.44411765, 0.44411765, 0.44264706,
       0.44264706, 0.44264706, 0.44264706, 0.44117647, 0.44117647,
       0.43823529, 0.43676471, 0.43529412, 0.43382353, 0.43235294,
       0.43088235, 0.42941176, 0.42941176, 0.42941176, 0.42941176,
       0.425     , 0.42205882, 0.41911765, 0.41764706, 0.41470588,
       0.41323529, 0.40588235, 0.40441176, 0.40441176, 0.40147059,
       0.4       , 0.39852941, 0.39264706, 0.39264706, 0.39264706,
       0.39117647, 0.38823529, 0.38676471, 0.38676471, 0.38382353,
       0.38235294, 0.38235294, 0.38088235, 0.37794118, 0.37647059,
       0.375     , 0.37058824, 0.36764706, 0.36617647, 0.36323529,
       0.35882353, 0.35735294, 0.35147059, 0.34558824, 0.34117647,
       0.33529412, 0.33235294, 0.32941176, 0.32352941, 0.31911765,
       0.31617647, 0.31470588, 0.31323529, 0.30441176, 0.3       ,
       0.28970588, 0.28529412, 0.28088235, 0.26911765, 0.25735294,
       0.24705882, 0.23529412, 0.22794118, 0.22352941, 0.21764706,
       0.21470588, 0.2       , 0.19558824, 0.18676471, 0.17794118,
       0.16617647, 0.15882353, 0.15      , 0.14411765, 0.13676471,
       0.13088235, 0.125     , 0.11764706, 0.11323529, 0.10588235,
       0.1       , 0.09117647, 0.08529412, 0.07794118, 0.06617647,
       0.05441176, 0.04705882, 0.02794118, 0.02058824, 0.01470588,
       0.00441176, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]

precition_cl_retain = [0.05059022, 0.05059022, 0.05059022, 0.05059022, 0.05059022,
       0.05575269, 0.07698048, 0.09860822, 0.12682813, 0.14603277,
       0.16111147, 0.17926014, 0.19649592, 0.21122631, 0.22513115,
       0.23802102, 0.25114737, 0.2628115 , 0.27974396, 0.29194346,
       0.30453726, 0.31656112, 0.32569257, 0.33541043, 0.343286  ,
       0.34910036, 0.35566331, 0.36224392, 0.3668088 , 0.37033619,
       0.37523777, 0.38102957, 0.38424256, 0.38944364, 0.39201762,
       0.39639778, 0.40099419, 0.40294359, 0.40732813, 0.411133  ,
       0.41328054, 0.41669927, 0.41924688, 0.42220488, 0.42654119,
       0.42877315, 0.43064411, 0.4331762 , 0.43386982, 0.43658247,
       0.43849326, 0.440522  , 0.44166175, 0.44430999, 0.4469696 ,
       0.44737258, 0.44800448, 0.4484324 , 0.45035303, 0.45109868,
       0.45137799, 0.45334198, 0.4551132 , 0.45627028, 0.45832352,
       0.45941597, 0.46042525, 0.46191615, 0.46434542, 0.46561539,
       0.46701137, 0.46714965, 0.46920607, 0.46966176, 0.47312593,
       0.47355131, 0.47670815, 0.47865029, 0.47847506, 0.48122096,
       0.48466689, 0.48852156, 0.49106549, 0.49321037, 0.49412244,
       0.49667274, 0.49663935, 0.49850433, 0.49960137, 0.50045628,
       0.50104382, 0.50110347, 0.50207349, 0.50197209, 0.50304163,
       0.50441484, 0.5074146 , 0.50841686, 0.50849548, 0.50935975,
       0.51237219, 0.51297644, 0.5128859 , 0.5149912 , 0.51583648,
       0.51843909, 0.52018221, 0.52157085, 0.5220438 , 0.52313169,
       0.52420688, 0.52529305, 0.52494937, 0.52495046, 0.52651227,
       0.53020409, 0.53089598, 0.53094197, 0.53511784, 0.53618947,
       0.54015049, 0.54186135, 0.54360385, 0.54676437, 0.55104248,
       0.55494105, 0.55781032, 0.55920011, 0.56334087, 0.565258  ,
       0.5659856 , 0.56786809, 0.5714784 , 0.57288137, 0.57525238,
       0.57607974, 0.57903623, 0.58145762, 0.58617879, 0.58950522,
       0.59154215, 0.59334024, 0.5931353 , 0.59852428, 0.60411522,
       0.61004255, 0.61288332, 0.61405339, 0.61568959, 0.61757344,
       0.62416433, 0.62744241, 0.62992175, 0.6322522 , 0.63351881,
       0.63881675, 0.64410195, 0.64438846, 0.64742568, 0.65098657,
       0.65238617, 0.65274488, 0.66106181, 0.66847799, 0.67010173,
       0.67170105, 0.68208641, 0.68808619, 0.69533136, 0.69853718,
       0.7062638 , 0.71394875, 0.72312421, 0.73414498, 0.73982304,
       0.75965653, 0.7573201 , 0.759597  , 0.76833485, 0.76765904,
       0.80640862, 0.81777003, 0.81391036, 0.8339308 , 0.89101307,
       0.90536381, 0.91043711, 0.92450142, 0.92411067, 0.92136223,
       0.92592593, 0.935     , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]

recall_cl_retain_death = [1.        , 1.        , 1.        , 1.        , 1.        ,
       0.988     , 0.95333333, 0.93466667, 0.90666667, 0.89733333,
       0.87866667, 0.85333333, 0.84266667, 0.83333333, 0.82533333,
       0.81733333, 0.79733333, 0.78266667, 0.776     , 0.76933333,
       0.75866667, 0.756     , 0.752     , 0.75066667, 0.74533333,
       0.744     , 0.74133333, 0.74      , 0.736     , 0.736     ,
       0.73333333, 0.73333333, 0.73066667, 0.73066667, 0.72666667,
       0.72666667, 0.72533333, 0.72133333, 0.72      , 0.71866667,
       0.71733333, 0.716     , 0.716     , 0.716     , 0.716     ,
       0.71466667, 0.71466667, 0.712     , 0.71066667, 0.70933333,
       0.708     , 0.70666667, 0.70533333, 0.70533333, 0.70533333,
       0.704     , 0.70266667, 0.69733333, 0.69733333, 0.696     ,
       0.696     , 0.69466667, 0.69333333, 0.69333333, 0.69333333,
       0.692     , 0.69066667, 0.69066667, 0.69066667, 0.68933333,
       0.68933333, 0.688     , 0.688     , 0.688     , 0.68666667,
       0.68533333, 0.684     , 0.684     , 0.68266667, 0.68266667,
       0.68266667, 0.68266667, 0.68266667, 0.68133333, 0.68      ,
       0.68      , 0.67866667, 0.67866667, 0.67866667, 0.67866667,
       0.67866667, 0.676     , 0.676     , 0.67466667, 0.672     ,
       0.672     , 0.672     , 0.672     , 0.67066667, 0.66933333,
       0.66933333, 0.66666667, 0.664     , 0.664     , 0.66266667,
       0.66266667, 0.66266667, 0.66266667, 0.66133333, 0.66133333,
       0.65866667, 0.65866667, 0.656     , 0.65333333, 0.652     ,
       0.652     , 0.65066667, 0.64933333, 0.64933333, 0.64666667,
       0.64266667, 0.64133333, 0.64      , 0.63866667, 0.63733333,
       0.63466667, 0.63333333, 0.63333333, 0.632     , 0.632     ,
       0.63066667, 0.62666667, 0.62666667, 0.62666667, 0.62533333,
       0.62533333, 0.624     , 0.62266667, 0.62133333, 0.62      ,
       0.61733333, 0.616     , 0.612     , 0.608     , 0.60533333,
       0.604     , 0.60266667, 0.60133333, 0.59866667, 0.59333333,
       0.59066667, 0.58933333, 0.58933333, 0.584     , 0.58      ,
       0.57733333, 0.57466667, 0.568     , 0.564     , 0.56      ,
       0.556     , 0.54666667, 0.54266667, 0.53733333, 0.53333333,
       0.524     , 0.51866667, 0.51066667, 0.504     , 0.488     ,
       0.476     , 0.46266667, 0.448     , 0.436     , 0.41466667,
       0.40133333, 0.38266667, 0.36      , 0.33733333, 0.30266667,
       0.28266667, 0.26      , 0.236     , 0.21333333, 0.19466667,
       0.17066667, 0.148     , 0.128     , 0.104     , 0.08      ,
       0.06      , 0.03466667, 0.00933333, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]

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

plt.plot(recall_ce_rnn_death_24, precision_ce_rnn_death, color='green', linestyle='dashed',linewidth=2, label='RNN+CE(AUC=0.823)')


plt.plot(recall_ce_death,precision_ce_death,color='blue',linestyle='dashed',linewidth=2,label='RETAIN+CE(AUC=0.828)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(precision_cl_rnn_death,recall_cl_rnn_death_24,color='violet',linewidth=1.5,label='RNN+CL(AUC=0.873)')
plt.plot(recall_cl_retain_death, precition_cl_retain, color='red', linewidth=1.5, label='RETAIN+CL(AUC=0.887)')


plt.legend(loc='lower right')
plt.show()
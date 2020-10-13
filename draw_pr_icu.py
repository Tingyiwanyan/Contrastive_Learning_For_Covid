import numpy as np
import matplotlib.pyplot as plt





precision_retain_ce = [0.16394208313872022,
 0.1640953716690042,
 0.1640953716690042,
 0.16455696202531644,
 0.16603595080416272,
 0.16843118383060635,
 0.1731815932706581,
 0.17662072485962227,
 0.18060735215769846,
 0.1887115165336374,
 0.19622181596587446,
 0.20155038759689922,
 0.2082758620689655,
 0.21674140508221226,
 0.23319502074688797,
 0.25279850746268656,
 0.28440366972477066,
 0.3685064935064935,
 0.5561497326203209,
 0.5843023255813954,
 0.6012084592145015,
 0.6018518518518519,
 0.6096774193548387,
 0.6125827814569537,
 0.6254295532646048,
 0.6459854014598541,
 0.6603053435114504,
 0.6693227091633466,
 0.6804979253112033,
 0.688034188034188,
 0.6855895196506551,
 0.6860986547085202,
 0.6930232558139535,
 0.7047619047619048,
 0.7156862745098039,
 0.7272727272727273,
 0.7282051282051282,
 0.7225130890052356,
 0.7311827956989247,
 0.7297297297297297,
 0.7417582417582418,
 0.75,
 0.7528089887640449,
 0.7701149425287356,
 0.7777777777777778,
 0.7771084337349398,
 0.7875,
 0.8104575163398693,
 0.8133333333333334,
 0.815068493150685,
 0.8129496402877698,
 0.8106060606060606,
 0.816793893129771,
 0.813953488372093,
 0.8174603174603174,
 0.8181818181818182,
 0.8389830508474576,
 0.8347826086956521,
 0.8392857142857143,
 0.8425925925925926,
 0.8476190476190476,
 0.8461538461538461,
 0.8461538461538461,
 0.8585858585858586,
 0.865979381443299,
 0.8617021276595744,
 0.8695652173913043,
 0.8695652173913043,
 0.8666666666666667,
 0.875,
 0.8780487804878049,
 0.8701298701298701,
 0.8918918918918919,
 0.8888888888888888,
 0.8840579710144928,
 0.8955223880597015,
 0.8939393939393939,
 0.890625,
 0.8852459016393442,
 0.8983050847457628,
 0.9056603773584906,
 0.9183673469387755,
 0.9148936170212766,
 0.9090909090909091,
 0.9285714285714286,
 0.9230769230769231,
 0.9210526315789473,
 0.9411764705882353,
 0.9375,
 0.9333333333333333,
 0.9629629629629629,
 0.96,
 0.9523809523809523,
 0.9523809523809523,
 0.9411764705882353,
 0.9090909090909091,
 0.875,
 0.8571428571428571,
 0.75,
 1.0,
 1]

recall_retain_ce = [1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 0.9971509971509972,
 0.9971509971509972,
 0.9857549857549858,
 0.9658119658119658,
 0.9430199430199431,
 0.9173789173789174,
 0.8888888888888888,
 0.8603988603988604,
 0.8262108262108262,
 0.8005698005698005,
 0.7720797720797721,
 0.7065527065527065,
 0.6467236467236467,
 0.5925925925925926,
 0.5726495726495726,
 0.5669515669515669,
 0.5555555555555556,
 0.5384615384615384,
 0.5270655270655271,
 0.5185185185185185,
 0.5042735042735043,
 0.4928774928774929,
 0.47863247863247865,
 0.4672364672364672,
 0.4586894586894587,
 0.4472934472934473,
 0.4358974358974359,
 0.42450142450142453,
 0.42165242165242167,
 0.41595441595441596,
 0.41025641025641024,
 0.4045584045584046,
 0.39316239316239315,
 0.38746438746438744,
 0.38461538461538464,
 0.38461538461538464,
 0.38461538461538464,
 0.3817663817663818,
 0.3817663817663818,
 0.3789173789173789,
 0.36752136752136755,
 0.358974358974359,
 0.35327635327635326,
 0.3475783475783476,
 0.33903133903133903,
 0.32193732193732194,
 0.30484330484330485,
 0.30484330484330485,
 0.29914529914529914,
 0.2934472934472934,
 0.28205128205128205,
 0.28205128205128205,
 0.27350427350427353,
 0.2678062678062678,
 0.25925925925925924,
 0.2535612535612536,
 0.25071225071225073,
 0.25071225071225073,
 0.24216524216524216,
 0.23931623931623933,
 0.23076923076923078,
 0.22792022792022792,
 0.22792022792022792,
 0.2222222222222222,
 0.21937321937321938,
 0.20512820512820512,
 0.1908831908831909,
 0.18803418803418803,
 0.18233618233618235,
 0.1737891737891738,
 0.17094017094017094,
 0.16809116809116809,
 0.1623931623931624,
 0.15384615384615385,
 0.150997150997151,
 0.13675213675213677,
 0.1282051282051282,
 0.1225071225071225,
 0.11396011396011396,
 0.1111111111111111,
 0.10256410256410256,
 0.09971509971509972,
 0.09116809116809117,
 0.08547008547008547,
 0.07977207977207977,
 0.07407407407407407,
 0.06837606837606838,
 0.05698005698005698,
 0.05698005698005698,
 0.045584045584045586,
 0.02849002849002849,
 0.019943019943019943,
 0.017094017094017096,
 0.008547008547008548,
 0.002849002849002849,
 0.0]

precision_ce_rnn_icu=[0.16394208313872022,
 0.16432584269662923,
 0.16486613433536873,
 0.16563834836260086,
 0.1680427391937834,
 0.17222222222222222,
 0.17866666666666667,
 0.18612521150592218,
 0.1925614877024595,
 0.20077469335054873,
 0.20882150241212957,
 0.22330827067669173,
 0.24299065420560748,
 0.2666015625,
 0.316043425814234,
 0.40098199672667756,
 0.4837310195227766,
 0.5501285347043702,
 0.5785123966942148,
 0.5914285714285714,
 0.6077844311377245,
 0.6238244514106583,
 0.6384364820846905,
 0.6438356164383562,
 0.6441281138790036,
 0.654275092936803,
 0.6666666666666666,
 0.6719367588932806,
 0.689795918367347,
 0.702928870292887,
 0.7229437229437229,
 0.7342342342342343,
 0.7361111111111112,
 0.7440758293838863,
 0.7548076923076923,
 0.7562189054726368,
 0.7537688442211056,
 0.7680412371134021,
 0.7708333333333334,
 0.7700534759358288,
 0.7692307692307693,
 0.776536312849162,
 0.7828571428571428,
 0.7906976744186046,
 0.7894736842105263,
 0.7869822485207101,
 0.7878787878787878,
 0.7901234567901234,
 0.7888198757763976,
 0.7884615384615384,
 0.7857142857142857,
 0.7814569536423841,
 0.7814569536423841,
 0.78,
 0.782312925170068,
 0.7972027972027972,
 0.7985611510791367,
 0.8029197080291971,
 0.8153846153846154,
 0.8110236220472441,
 0.8048780487804879,
 0.8099173553719008,
 0.8151260504201681,
 0.808695652173913,
 0.8288288288288288,
 0.8288288288288288,
 0.8333333333333334,
 0.8476190476190476,
 0.8415841584158416,
 0.8383838383838383,
 0.8367346938775511,
 0.84375,
 0.8478260869565217,
 0.8505747126436781,
 0.8690476190476191,
 0.875,
 0.8701298701298701,
 0.863013698630137,
 0.8695652173913043,
 0.8769230769230769,
 0.8870967741935484,
 0.8852459016393442,
 0.8852459016393442,
 0.896551724137931,
 0.8947368421052632,
 0.8888888888888888,
 0.8823529411764706,
 0.875,
 0.8666666666666667,
 0.8636363636363636,
 0.8480487804878049,
 0.868918918918919,
 0.91875,
 0.9142857142857143,
 0.9129629629629629,
 0.9045454545454546,
 0.8944444444444444,
 0.8885714285714286,
 0.87,
 0.8333333333333334,
 1]

recall_ce_rnn_icu=[1.0,
 1.0,
 1.0,
 0.9943019943019943,
 0.9857549857549858,
 0.9715099715099715,
 0.9544159544159544,
 0.9401709401709402,
 0.9145299145299145,
 0.886039886039886,
 0.8632478632478633,
 0.8461538461538461,
 0.8148148148148148,
 0.7777777777777778,
 0.7464387464387464,
 0.698005698005698,
 0.6353276353276354,
 0.6096866096866097,
 0.5982905982905983,
 0.5897435897435898,
 0.5783475783475783,
 0.5669515669515669,
 0.5584045584045584,
 0.5356125356125356,
 0.5156695156695157,
 0.5014245014245015,
 0.49002849002849,
 0.4843304843304843,
 0.48148148148148145,
 0.47863247863247865,
 0.4757834757834758,
 0.46438746438746437,
 0.452991452991453,
 0.4472934472934473,
 0.4472934472934473,
 0.43304843304843305,
 0.42735042735042733,
 0.42450142450142453,
 0.42165242165242167,
 0.41025641025641024,
 0.39886039886039887,
 0.396011396011396,
 0.3903133903133903,
 0.38746438746438744,
 0.38461538461538464,
 0.3789173789173789,
 0.37037037037037035,
 0.3646723646723647,
 0.36182336182336183,
 0.3504273504273504,
 0.34472934472934474,
 0.33618233618233617,
 0.33618233618233617,
 0.3333333333333333,
 0.32763532763532766,
 0.3247863247863248,
 0.3162393162393162,
 0.31339031339031337,
 0.301994301994302,
 0.2934472934472934,
 0.28205128205128205,
 0.2792022792022792,
 0.27635327635327633,
 0.26495726495726496,
 0.2621082621082621,
 0.2621082621082621,
 0.2564102564102564,
 0.2535612535612536,
 0.24216524216524216,
 0.23646723646723647,
 0.2336182336182336,
 0.23076923076923078,
 0.2222222222222222,
 0.21082621082621084,
 0.20797720797720798,
 0.19943019943019943,
 0.1908831908831909,
 0.1794871794871795,
 0.17094017094017094,
 0.1623931623931624,
 0.15669515669515668,
 0.15384615384615385,
 0.15384615384615385,
 0.14814814814814814,
 0.1452991452991453,
 0.13675213675213677,
 0.1282051282051282,
 0.11965811965811966,
 0.1111111111111111,
 0.10826210826210826,
 0.10256410256410256,
 0.09686609686609686,
 0.08831908831908832,
 0.07692307692307693,
 0.07407407407407407,
 0.05982905982905983,
 0.04843304843304843,
 0.037037037037037035,
 0.02564102564102564,
 0.014245014245014245,
 0.0]

precision_RNN_cl = [0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16532829475673122,
 0.2164723032069971,
 0.2547993019197208,
 0.28484231943031535,
 0.3098751418842225,
 0.3279503105590062,
 0.341688654353562,
 0.3615819209039548,
 0.3702064896755162,
 0.3878504672897196,
 0.4019607843137255,
 0.4114671163575042,
 0.41924398625429554,
 0.42857142857142855,
 0.4358047016274864,
 0.44220183486238535,
 0.450281425891182,
 0.4605009633911368,
 0.462890625,
 0.466403162055336,
 0.47105788423153694,
 0.4729458917835671,
 0.47484909456740443,
 0.48459958932238195,
 0.48760330578512395,
 0.49166666666666664,
 0.49473684210526314,
 0.4968152866242038,
 0.5021459227467812,
 0.5054229934924078,
 0.5088105726872246,
 0.5088888888888888,
 0.5123042505592841,
 0.5180995475113123,
 0.5204545454545455,
 0.5205479452054794,
 0.5253456221198156,
 0.5279069767441861,
 0.5279069767441861,
 0.531615925058548,
 0.531615925058548,
 0.5306603773584906,
 0.5296912114014252,
 0.5322195704057279,
 0.5334928229665071,
 0.5336538461538461,
 0.5351089588377724,
 0.537712895377129,
 0.5390243902439025,
 0.5443349753694581,
 0.5432098765432098,
 0.5445544554455446,
 0.5447761194029851,
 0.5475,
 0.5488721804511278,
 0.5521628498727735,
 0.5535714285714286,
 0.5541237113402062,
 0.5555555555555556,
 0.5572916666666666,
 0.5602094240837696,
 0.5590551181102362,
 0.5608465608465608,
 0.5608465608465608,
 0.5638297872340425,
 0.5653333333333334,
 0.5653333333333334,
 0.5683646112600537,
 0.5683646112600537,
 0.5683646112600537,
 0.5698924731182796,
 0.5714285714285714,
 0.5745257452574526,
 0.5760869565217391,
 0.5760869565217391,
 0.5760869565217391,
 0.5776566757493188,
 0.5792349726775956,
 0.5792349726775956,
 0.5808219178082191,
 0.5824175824175825,
 0.581267217630854,
 0.5817174515235457,
 0.5821727019498607,
 0.5837988826815642,
 0.5854341736694678,
 0.5870786516853933,
 0.5920679886685553,
 0.5954415954415955,
 0.5977011494252874,
 0.5977011494252874,
 0.5977011494252874,
 0.6,
 0.6005830903790087,
 0.6005917159763313,
 0.6006006006006006,
 0.603030303030303,
 0.6085626911314985,
 0.6123076923076923,
 0.6160990712074303,
 0.6180124223602484,
 0.61875,
 0.6214511041009464,
 0.6234177215189873,
 0.6282051282051282,
 0.632258064516129,
 0.6310679611650486,
 0.6331168831168831,
 0.6319218241042345,
 0.6339869281045751,
 0.6360655737704918,
 0.636963696369637,
 0.636963696369637,
 0.636963696369637,
 0.636963696369637,
 0.6421404682274248,
 0.6452702702702703,
 0.6484641638225256,
 0.6484641638225256,
 0.6484641638225256,
 0.6482758620689655,
 0.6482758620689655,
 0.6515679442508711,
 0.6607773851590106,
 0.6594982078853047,
 0.6642599277978339,
 0.6690909090909091,
 0.6678832116788321,
 0.6691176470588235,
 0.6691176470588235,
 0.6715867158671587,
 0.674074074074074,
 0.6765799256505576,
 0.6816479400749064,
 0.6893939393939394,
 0.6923076923076923,
 0.6937984496124031,
 0.6941176470588235,
 0.6996047430830039,
 0.7016129032258065,
 0.7032520325203252,
 0.7119341563786008,
 0.7095435684647303,
 0.7083333333333334,
 0.7088607594936709,
 0.7106382978723405,
 0.7130434782608696,
 0.7161572052401747,
 0.7136563876651982,
 0.71875,
 0.7149321266968326,
 0.7142857142857143,
 0.7129629629629629,
 0.7109004739336493,
 0.7095238095238096,
 0.7087378640776699,
 0.7121951219512195,
 0.7114427860696517,
 0.7106598984771574,
 0.7061855670103093,
 0.7120418848167539,
 0.708994708994709,
 0.7081081081081081,
 0.7111111111111111,
 0.7045454545454546,
 0.7011494252873564,
 0.7041420118343196,
 0.7048192771084337,
 0.7018633540372671,
 0.7088607594936709,
 0.7105263157894737,
 0.7092198581560284,
 0.7142857142857143,
 0.7307692307692307,
 0.7377049180327869,
 0.7350427350427351,
 0.7657657657657657,
 0.77,
 0.7912087912087912,
 0.7738095238095238,
 0.7714285714285715,
 0.7962962962962963,
 0.8,
 0.8114285714285715,
 0.783333333333333,
 0.81,
 0.85,
 0.915,
 0.8333333333333334,
 1.0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1]

recall_cl_rnn = [1.0,
 1.0,
 1.0,
 1.0,
 0.9971509971509972,
 0.8461538461538461,
 0.8319088319088319,
 0.7977207977207977,
 0.7777777777777778,
 0.7521367521367521,
 0.7378917378917379,
 0.7293447293447294,
 0.7150997150997151,
 0.7094017094017094,
 0.7008547008547008,
 0.6951566951566952,
 0.6951566951566952,
 0.6923076923076923,
 0.6866096866096866,
 0.6866096866096866,
 0.6837606837606838,
 0.6809116809116809,
 0.6752136752136753,
 0.6723646723646723,
 0.6723646723646723,
 0.6723646723646723,
 0.6723646723646723,
 0.6723646723646723,
 0.6723646723646723,
 0.6723646723646723,
 0.6695156695156695,
 0.6666666666666666,
 0.6666666666666666,
 0.6638176638176638,
 0.6581196581196581,
 0.6524216524216524,
 0.6524216524216524,
 0.6524216524216524,
 0.6524216524216524,
 0.6495726495726496,
 0.6495726495726496,
 0.6467236467236467,
 0.6467236467236467,
 0.6467236467236467,
 0.6467236467236467,
 0.6410256410256411,
 0.6353276353276354,
 0.6353276353276354,
 0.6353276353276354,
 0.6324786324786325,
 0.6296296296296297,
 0.6296296296296297,
 0.6296296296296297,
 0.6296296296296297,
 0.6267806267806267,
 0.6267806267806267,
 0.6239316239316239,
 0.6239316239316239,
 0.6239316239316239,
 0.6182336182336182,
 0.6182336182336182,
 0.6125356125356125,
 0.6125356125356125,
 0.6096866096866097,
 0.6096866096866097,
 0.6068376068376068,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.603988603988604,
 0.6011396011396012,
 0.5982905982905983,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5925925925925926,
 0.5925925925925926,
 0.5925925925925926,
 0.5897435897435898,
 0.5868945868945868,
 0.5783475783475783,
 0.5698005698005698,
 0.5669515669515669,
 0.5669515669515669,
 0.5669515669515669,
 0.5669515669515669,
 0.5669515669515669,
 0.5641025641025641,
 0.5612535612535613,
 0.5612535612535613,
 0.5584045584045584,
 0.5584045584045584,
 0.5555555555555556,
 0.5555555555555556,
 0.5527065527065527,
 0.5527065527065527,
 0.5527065527065527,
 0.5498575498575499,
 0.5498575498575499,
 0.5498575498575499,
 0.5498575498575499,
 0.5470085470085471,
 0.5441595441595442,
 0.5413105413105413,
 0.5413105413105413,
 0.5413105413105413,
 0.5356125356125356,
 0.5356125356125356,
 0.5327635327635327,
 0.5327635327635327,
 0.5242165242165242,
 0.5242165242165242,
 0.5242165242165242,
 0.5213675213675214,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5128205128205128,
 0.50997150997151,
 0.5042735042735043,
 0.5042735042735043,
 0.49572649572649574,
 0.4928774928774929,
 0.4928774928774929,
 0.48717948717948717,
 0.4843304843304843,
 0.47863247863247865,
 0.4757834757834758,
 0.4672364672364672,
 0.4672364672364672,
 0.46153846153846156,
 0.4586894586894587,
 0.45014245014245013,
 0.4415954415954416,
 0.43874643874643876,
 0.42735042735042733,
 0.42450142450142453,
 0.41595441595441596,
 0.41595441595441596,
 0.4074074074074074,
 0.39886039886039887,
 0.3903133903133903,
 0.38746438746438744,
 0.3817663817663818,
 0.3732193732193732,
 0.3646723646723647,
 0.35327635327635326,
 0.3475783475783476,
 0.33903133903133903,
 0.3333333333333333,
 0.32193732193732194,
 0.3190883190883191,
 0.3076923076923077,
 0.2849002849002849,
 0.2706552706552707,
 0.2706552706552707,
 0.2564102564102564,
 0.245014245014245,
 0.24216524216524216,
 0.21937321937321938,
 0.20512820512820512,
 0.18518518518518517,
 0.15384615384615385,
 0.1225071225071225,
 0.10256410256410256,
 0.07692307692307693,
 0.06267806267806268,
 0.037037037037037035,
 0.02564102564102564,
 0.019943019943019943,
 0.014245014245014245,
 0.011396011396011397,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]


precision_cl_retain_icu = [0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.16394208313872022,
 0.1640953716690042,
 0.16424894712213384,
 0.1643192488262911,
 0.2395498392282958,
 0.2480948348856901,
 0.27680311890838205,
 0.31942789034564956,
 0.36633663366336633,
 0.4,
 0.4172297297297297,
 0.43189964157706096,
 0.44612476370510395,
 0.45866141732283466,
 0.46356275303643724,
 0.47401247401247404,
 0.48394004282655245,
 0.49130434782608695,
 0.49667405764966743,
 0.49887640449438203,
 0.5034324942791762,
 0.5081206496519721,
 0.5116822429906542,
 0.5141509433962265,
 0.5157384987893463,
 0.5208845208845209,
 0.5260545905707196,
 0.5288220551378446,
 0.5316455696202531,
 0.5343511450381679,
 0.5331632653061225,
 0.5345268542199488,
 0.538659793814433,
 0.5428571428571428,
 0.5471204188481675,
 0.5488126649076517,
 0.5517241379310345,
 0.553475935828877,
 0.5540540540540541,
 0.5555555555555556,
 0.5570652173913043,
 0.5589041095890411,
 0.5589041095890411,
 0.5580110497237569,
 0.5586592178770949,
 0.5617977528089888,
 0.5633802816901409,
 0.5633802816901409,
 0.5625,
 0.56,
 0.56,
 0.5616045845272206,
 0.5606936416184971,
 0.5606936416184971,
 0.561046511627907,
 0.5630498533724341,
 0.5667655786350149,
 0.5667655786350149,
 0.5684523809523809,
 0.5684523809523809,
 0.5735735735735735,
 0.5735735735735735,
 0.5757575757575758,
 0.5775075987841946,
 0.5792682926829268,
 0.5828220858895705,
 0.5846153846153846,
 0.5820433436532507,
 0.5820433436532507,
 0.5820433436532507,
 0.5838509316770186,
 0.5856697819314641,
 0.5856697819314641,
 0.5867507886435331,
 0.5904761904761905,
 0.5923566878980892,
 0.5942492012779552,
 0.592948717948718,
 0.5967741935483871,
 0.5954692556634305,
 0.5954692556634305,
 0.5993485342019544,
 0.6052631578947368,
 0.6052631578947368,
 0.6059602649006622,
 0.6079734219269103,
 0.61,
 0.61,
 0.6148648648648649,
 0.6169491525423729,
 0.621160409556314,
 0.6219931271477663,
 0.6223776223776224,
 0.6289752650176679,
 0.6312056737588653,
 0.6321428571428571,
 0.6344086021505376,
 0.6353790613718412,
 0.6363636363636364,
 0.6363636363636364,
 0.6397058823529411,
 0.6444444444444445,
 0.6455223880597015,
 0.6479400749063671,
 0.6479400749063671,
 0.650375939849624,
 0.6515151515151515,
 0.6590038314176245,
 0.6550387596899225,
 0.6614173228346457,
 0.6600790513833992,
 0.6666666666666666,
 0.6693548387096774,
 0.6748971193415638,
 0.6778242677824268,
 0.6822033898305084,
 0.6851063829787234,
 0.6810344827586207,
 0.6785714285714286,
 0.6788990825688074,
 0.6790697674418604,
 0.6807511737089202,
 0.680952380952381,
 0.6844660194174758,
 0.6865671641791045,
 0.6938775510204082,
 0.7096774193548387,
 0.7039106145251397,
 0.7048192771084337,
 0.7012987012987013,
 0.6838235294117647,
 0.6727272727272727,
 0.6951219512195121,
 0.7166666666666667,
 0.5666666666666667,
 0.6666666666666666,
 0.5,
 0.5,
 1.0,
 1.0,
 1.0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1]

recall_cl_retain_icu = [1.0,
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
 0.9971509971509972,
 0.8490028490028491,
 0.8347578347578347,
 0.8091168091168092,
 0.7635327635327636,
 0.7378917378917379,
 0.717948717948718,
 0.7037037037037037,
 0.6866096866096866,
 0.6723646723646723,
 0.6638176638176638,
 0.6524216524216524,
 0.6495726495726496,
 0.6438746438746439,
 0.6438746438746439,
 0.6381766381766382,
 0.6324786324786325,
 0.6267806267806267,
 0.6239316239316239,
 0.6239316239316239,
 0.6210826210826211,
 0.6068376068376068,
 0.603988603988604,
 0.603988603988604,
 0.6011396011396012,
 0.5982905982905983,
 0.5982905982905983,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5954415954415955,
 0.5925925925925926,
 0.5925925925925926,
 0.5897435897435898,
 0.584045584045584,
 0.584045584045584,
 0.584045584045584,
 0.5811965811965812,
 0.5811965811965812,
 0.5754985754985755,
 0.5698005698005698,
 0.5698005698005698,
 0.5698005698005698,
 0.5698005698005698,
 0.5641025641025641,
 0.5584045584045584,
 0.5584045584045584,
 0.5584045584045584,
 0.5527065527065527,
 0.5527065527065527,
 0.5498575498575499,
 0.5470085470085471,
 0.5441595441595442,
 0.5441595441595442,
 0.5441595441595442,
 0.5441595441595442,
 0.5441595441595442,
 0.5441595441595442,
 0.5413105413105413,
 0.5413105413105413,
 0.5413105413105413,
 0.5413105413105413,
 0.5413105413105413,
 0.5356125356125356,
 0.5356125356125356,
 0.5356125356125356,
 0.5356125356125356,
 0.5356125356125356,
 0.5356125356125356,
 0.5299145299145299,
 0.5299145299145299,
 0.5299145299145299,
 0.5299145299145299,
 0.5270655270655271,
 0.5270655270655271,
 0.5242165242165242,
 0.5242165242165242,
 0.5242165242165242,
 0.5242165242165242,
 0.5242165242165242,
 0.5213675213675214,
 0.5213675213675214,
 0.5213675213675214,
 0.5213675213675214,
 0.5185185185185185,
 0.5185185185185185,
 0.5185185185185185,
 0.5156695156695157,
 0.5071225071225072,
 0.5071225071225072,
 0.5071225071225072,
 0.5042735042735043,
 0.5042735042735043,
 0.5014245014245015,
 0.4985754985754986,
 0.4985754985754986,
 0.49572649572649574,
 0.49572649572649574,
 0.4928774928774929,
 0.4928774928774929,
 0.4928774928774929,
 0.4928774928774929,
 0.49002849002849,
 0.49002849002849,
 0.48148148148148145,
 0.47863247863247865,
 0.4757834757834758,
 0.47293447293447294,
 0.47293447293447294,
 0.4672364672364672,
 0.46153846153846156,
 0.4586894586894587,
 0.4586894586894587,
 0.45014245014245013,
 0.43304843304843305,
 0.42165242165242167,
 0.41595441595441596,
 0.4131054131054131,
 0.4074074074074074,
 0.4017094017094017,
 0.39316239316239315,
 0.38746438746438744,
 0.37606837606837606,
 0.358974358974359,
 0.3333333333333333,
 0.3076923076923077,
 0.26495726495726496,
 0.21082621082621084,
 0.1623931623931624,
 0.1225071225071225,
 0.04843304843304843,
 0.022792022792022793,
 0.008547008547008548,
 0.008547008547008548,
 0.008547008547008548,
 0.005698005698005698,
 0.002849002849002849,
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
 0.0,
 0.0]




plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mortality Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(recall_ce_rnn_icu, precision_ce_rnn_icu, color='green', linewidth=1, label='RNN+CE(AUC=0.875)')


plt.plot(recall_retain_ce,precision_retain_ce,color='blue',label='RETAIN+CE(AUC=0.888)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(recall_cl_rnn,precision_RNN_cl,color='violet',label='RNN+CL(AUC=0.928)')
plt.plot(recall_cl_retain_icu, precision_cl_retain_icu, color='red', linewidth=1, label='RETAIN+CL(AUC=0.913)')


plt.legend(loc='lower right')
plt.show()
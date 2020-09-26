import numpy as np
import random
import math
import time
import pandas as pd
import json
from LSTM import LSTM_model
from Data_process import kg_process_data
from Dynamic_HGM import dynamic_hgm
from MLP import MLP_model


class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """

    def __init__(self):
        file_path = '/datadrive/tingyi_wanyan/user_tingyi.wanyan/tensorflow_venv/registry_2020-06-29'
        self.reg = file_path + '/registry.csv'
        self.covid_lab = file_path + '/covid19LabTest.csv'
        self.lab = file_path + '/Lab.csv'
        self.vital = file_path + '/vitals.csv'
        file_path_ = '/home/tingyi.wanyan'
        self.lab_comb = 'lab_mapping_comb.csv'
        self.file_path_comorbidity = '/home/tingyi.wanyan/comorbidity_matrix_20200710.csv'

    def read_csv(self):
        self.registry = pd.read_csv(self.reg)
        self.covid_labtest = pd.read_csv(self.covid_lab)
        self.labtest = pd.read_csv(self.lab)
        self.vital_sign = pd.read_csv(self.vital)
        # self.comorbidity = pd.read_csv(self.file_path_comorbidity)
        self.lab_comb = pd.read_csv(self.lab_comb)
        self.reg_ar = np.array(self.registry)
        self.covid_ar = np.array(self.covid_labtest)
        self.labtest_ar = np.array(self.labtest)
        self.vital_sign_ar = np.array(self.vital_sign)
        self.lab_comb_ar = np.array(self.lab_comb)

    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_vital = {}
        self.dic_lab = {}
        self.dic_lab_category = {}
        self.dic_demographic = {}
        self.dic_race = {}
        self.crucial_vital = ['CAC - BLOOD PRESSURE', 'CAC - TEMPERATURE', 'CAC - PULSE OXIMETRY',
                              'CAC - RESPIRATIONS', 'CAC - PULSE', 'CAC - HEIGHT', 'CAC - WEIGHT/SCALE']
        index_keep = np.where(self.lab_comb_ar[:, -1] == 1)[0]
        self.lab_comb_keep = self.lab_comb_ar[index_keep]
        index_name = np.where(self.lab_comb_keep[:, -2] == self.lab_comb_keep[:, -2])[0]
        self.lab_test_feature = []
        [self.lab_test_feature.append(i) for i in self.lab_comb_keep[:, -2] if i not in self.lab_test_feature]
        self.lab_comb_keep_ = self.lab_comb_keep[index_name]
        self.cat_comb = self.lab_comb_keep[:, [0, -2]]
        """
        create inital lab dictionary
        """
        index_lab = 0
        for i in range(index_name.shape[0]):
            name_test = self.lab_comb_keep[i][0]
            name_category = self.lab_comb_keep[i][-2]
            if name_test not in self.dic_lab_category.keys():
                self.dic_lab_category[name_test] = name_category
                if name_category not in self.dic_lab:
                    self.dic_lab[name_category] = {}
                    # self.dic_lab[name_category]['patient_values'] = {}
                    # self.dic_lan[name_category]['specific name']={}
                    # self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
                    self.dic_lab[name_category]['index'] = index_lab
                    index_lab += 1
                # else:
                #   self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
        """
        create initial vital sign dictionary
        """
        index_vital = 0
        for i in self.crucial_vital:
            if i == 'CAC - BLOOD PRESSURE':
                self.dic_vital['high'] = {}
                self.dic_vital['high']['index'] = index_vital
                index_vital += 1
                self.dic_vital['low'] = {}
                self.dic_vital['low']['index'] = index_vital
                index_vital += 1
            else:
                self.dic_vital[i] = {}
                self.dic_vital[i]['index'] = index_vital
                index_vital += 1

        icu = np.where(self.reg_ar[:, 29] == self.reg_ar[:, 29])[0]
        for i in icu:
            mrn_single = self.reg_ar[i, 45]
            in_time_single = self.reg_ar[i, 29]
            if self.reg_ar[i, 11] == self.reg_ar[i, 11]:
                death_flag = 1
            else:
                death_flag = 0
            # self.dic_patient[mrn_single]={}
            # self.dic_patient[mrn_single]['in_icu_time']=in_time_single
            # self.dic_patient[mrn_single]['death_flag']=death_flag
            self.in_time = in_time_single.split(' ')
            in_date = [np.int(i) for i in self.in_time[0].split('-')]
            in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
            self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
            in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
            total_in_time_value = in_date_value + in_time_value
            if mrn_single not in self.dic_patient.keys():
                self.dic_patient[mrn_single] = {}
                self.dic_patient[mrn_single]['in_icu_time'] = self.in_time
                self.dic_patient[mrn_single]['in_date'] = in_date
                self.dic_patient[mrn_single]['in_time'] = self.in_time_
                self.dic_patient[mrn_single]['death_flag'] = death_flag
                self.dic_patient[mrn_single]['total_in_time_value'] = total_in_time_value
                self.dic_patient[mrn_single]['prior_time_vital'] = {}
                self.dic_patient[mrn_single]['prior_time_lab'] = {}
                # self.dic_patient[mrn_single]['lab_time_check']={}
                # self.dic_patient[mrn_single]['time_capture']={}
        mrn_icu = self.reg_ar[:, 45][icu]
        covid_detect = np.where(self.covid_ar[:, 7] != 'NOT DETECTED')[0]
        covid_mrn = self.covid_ar[:, 0][covid_detect]
        self.total_data = np.intersect1d(list(covid_mrn), list(mrn_icu))
        index = 0
        for i in self.dic_lab.keys():
            test_specific = self.lab_comb_keep_[np.where(self.lab_comb_keep_[:, -2] == i)[0]][:, 0]
            num = 0
            test_patient_specific = []
            for j in test_specific:
                test_patient_specific += list(self.labtest_ar[np.where(kg.labtest_ar[:, 2] == j)[0]][:, 0])
            num += len(np.intersect1d(list(test_patient_specific), self.total_data))
            self.dic_lab[i]['num_patient'] = num

        """
        exclude those not sure whether discharge or death
        """
        total_data_check = []
        for i in self.total_data:
            check_icu = np.where(self.reg_ar[:, 45] == i)[0]
            for j in check_icu:
                if self.reg_ar[j][11] == self.reg_ar[j][11]:
                    if i not in total_data_check:
                        total_data_check.append(i)
                elif self.reg_ar[j][30] == self.reg_ar[j][30]:
                    if i not in total_data_check:
                        total_data_check.append(i)
                else:
                    continue
        self.total_data = total_data_check
        index_race = 0
        for i in self.total_data:
            index_race_ = np.where(self.reg_ar[:, 45] == i)[0]
            self.check_index = index_race_
            race = 0
            for j in index_race_:
                race_check = self.reg_ar[:, 61][j]
                if race_check == race_check:
                    race = race_check
                    break
            for j in index_race_:
                age_check = self.reg_ar[:, 7][j]
                if age_check == age_check:
                    age = age_check
                    break
            for j in index_race_:
                gender_check = self.reg_ar[:, 24][j]
                if gender_check == gender_check:
                    gender = gender_check
                    break
            # self.dic_race['Age']=age
            # self.dic_race['gender']=gender
            if race == 0:
                continue
            if race[0] == 'A':
                if 'A' not in self.dic_race:
                    self.dic_race['A'] = {}
                    self.dic_race['A']['num'] = 1
                    self.dic_race['A']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['A']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'A'
            elif race[0] == 'B':
                if 'B' not in self.dic_race:
                    self.dic_race['B'] = {}
                    self.dic_race['B']['num'] = 1
                    self.dic_race['B']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['B']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'B'
            elif race[0] == '<':
                race_ = race.split('>')[3].split('<')[0]
                if race_ not in self.dic_race:
                    self.dic_race[race_] = {}
                    self.dic_race[race_]['num'] = 1
                    self.dic_race[race_]['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race[race_]['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = race_
            elif race[0] == 'I' or race[0] == 'P':
                if 'U' not in self.dic_race:
                    self.dic_race['U'] = {}
                    self.dic_race['U']['num'] = 1
                    self.dic_race['U']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['U']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'U'
            else:
                if race not in self.dic_race:
                    self.dic_race[race] = {}
                    self.dic_race[race]['num'] = 1
                    self.dic_race[race]['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race[race]['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = race
            if 'Age' not in self.dic_race:
                self.dic_race['Age'] = {}
                self.dic_race['Age']['index'] = index_race
                index_race += 1
            self.dic_demographic[i]['Age'] = age
            # index_race += 1
            if 'M' not in self.dic_race:
                self.dic_race['M'] = {}
                self.dic_race['M']['index'] = index_race
                index_race += 1
            if 'F' not in self.dic_race:
                self.dic_race['F'] = {}
                self.dic_race['F']['index'] = index_race
                index_race += 1
            self.dic_demographic[i]['gender'] = gender

        for i in self.total_data:
            in_icu_date = self.reg_ar
            self.single_patient_vital = np.where(self.vital_sign_ar[:, 0] == i)[0]
            in_time_value = self.dic_patient[i]['total_in_time_value']
            self.single_patient_lab = np.where(self.labtest_ar[:, 0] == i)[0]
            total_value_lab = 0

            for k in self.single_patient_lab:
                obv_id = self.labtest_ar[k][2]
                patient_lab_mrn = self.labtest_ar[k][0]
                value = self.labtest_ar[k][3]
                self.check_data_lab = self.labtest_ar[k][4]
                date_year_value_lab = float(str(self.labtest_ar[k][4])[0:4]) * 365
                date_day_value_lab = float(str(self.check_data_lab)[4:6]) * 30 + float(str(self.check_data_lab)[6:8])
                date_value_lab = (date_year_value_lab + date_day_value_lab) * 24 * 60
                date_time_value_lab = float(str(self.check_data_lab)[8:10]) * 60 + float(
                    str(self.check_data_lab)[10:12])
                self.total_time_value_lab = date_value_lab + date_time_value_lab
                self.dic_patient[i].setdefault('lab_time_check', []).append(self.check_data_lab)
                if obv_id in self.dic_lab_category.keys():
                    category = self.dic_lab_category[obv_id]
                    self.prior_time = np.int(np.floor(np.float((self.total_time_value_lab - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    try:
                        value = float(value)
                    except:
                        continue
                    if not value == value:
                        continue
                    if i not in self.dic_lab[category]:
                        # self.dic_lab[category]['patient_values'][i]={}
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    else:
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    if self.prior_time not in self.dic_patient[i]['prior_time_lab']:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time] = {}
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)
                    else:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)
            # if not self.dic_lab[category]['patient_values'][i] == {}:
            #   mean_value_lab_single = np.mean(self.dic_lab[category]['patient_values'][i]['lab_value_patient'])
            #  self.dic_lab[category]['patient_values'][i]['lab_mean_value']=mean_value_lab_single

            # print(index)
            # index += 1
            for j in self.single_patient_vital:
                obv_id = self.vital_sign_ar[j][2]
                if obv_id in self.crucial_vital:
                    self.check_data = self.vital_sign_ar[j][4]
                    self.dic_patient[i].setdefault('time_capture', []).append(self.check_data)
                    date_year_value = float(str(self.vital_sign_ar[j][4])[0:4]) * 365
                    date_day_value = float(str(self.check_data)[4:6]) * 30 + float(str(self.check_data)[6:8])
                    date_value = (date_year_value + date_day_value) * 24 * 60
                    date_time_value = float(str(self.check_data)[8:10]) * 60 + float(str(self.check_data)[10:12])
                    total_time_value = date_value + date_time_value
                    self.prior_time = np.int(np.floor(np.float((total_time_value - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    if obv_id == 'CAC - BLOOD PRESSURE':
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        self.check_value_presure = self.vital_sign_ar[j][3]
                        try:
                            value = self.vital_sign_ar[j][3].split('/')
                        except:
                            continue
                        if self.check_value_presure == '""':
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                value[1])
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                value[1])
                        self.dic_vital['high'].setdefault('value', []).append(value[0])
                        self.dic_vital['low'].setdefault('value', []).append(value[1])
                    else:
                        self.check_value = self.vital_sign_ar[j][3]
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        if self.check_value == '""':
                            continue
                        value = float(self.vital_sign_ar[j][3])
                        if np.isnan(value):
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        self.dic_vital[obv_id].setdefault('value', []).append(value)
        """
        for i in self.dic_lab.keys():
            mean_value_total = []
            length_patient_lab = len(list(self.dic_lab[i].keys()))-1
            #mean_value_single_lab = 0
            for j in self.dic_lab[i]['patient_values'].keys():
                #mean_value_single_lab += self.dic_lab[i]['patient_values'][j]['lab_mean_value']
                if not self.dic_lab[i]['patient_values'][j] == {}:
                    mean_value_total.append(self.dic_lab[i]['patient_values'][j]['lab_mean_value'])
            #mean_value_lab = float(mean_value_single_lab)/length_patient_lab
            mean_value_lab = np.mean(mean_value_total)
            std_lab = np.std(mean_value_total)
            self.dic_lab[i]['mean_value'] = mean_value_lab
            self.dic_lab[i]['std'] = std_lab

        res = []        
        [res.append(x) for x in list(self.labtest_ar[:,2]) if x not in res]
        res_cur = []
        [res_cur.append(x) for x in res if x==x]
        for i in range(len(res_cur)):
            num = np.intersect1d(list(kg.labtest_ar[np.where(kg.labtest_ar[:,2]==res_cur[i])[0]][:,0]),list(kg.total_data)).shape[0]
            res_cur[i] = res_cur[i]+' ' + str(num)
        """


if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.read_csv()
    # kg.create_kg_dic()

    for i in kg.dic_lab.keys():
        mean_lab = np.mean(kg.dic_lab[i]['lab_value_patient'])
        std_lab = np.mean(kg.dic_lab[i]['lab_value_patient'])
        kg.dic_lab[i]['mean_value'] = mean_lab
        kg.dic_lab[i]['std'] = std_lab

    for i in kg.dic_vital.keys():
        values = [np.float(j) for j in kg.dic_vital[i]['value']]
        mean = np.mean(values)
        std = np.std(values)
        kg.dic_vital[i]['mean_value'] = mean
        kg.dic_vital[i]['std'] = std
    """
    kg.dic_death = {}
    for i in kg.total_data:
        if kg.dic_patient[i]['death_flag'] == 0:
            kg.dic_death.setdefault(0,[]).append(i)
        if kg.dic_patient[i]['death_flag'] == 1:
            kg.dic_death.setdefault(1,[]).append(i)
    """
    total_data_check = []
    for i in kg.total_data:
        check_icu = np.where(kg.reg_ar[:, 45] == i)[0]
        for j in check_icu:
            if kg.reg_ar[j][11] == kg.reg_ar[j][11]:
                if i not in total_data_check:
                    total_data_check.append(i)
            elif kg.reg_ar[j][30] == kg.reg_ar[j][30]:
                if i not in total_data_check:
                    total_data_check.append(i)
            else:
                continue
    kg.total_data = total_data_check
    kg.dic_death = {}
    for i in kg.total_data:
        if kg.dic_patient[i]['death_flag'] == 0:
            kg.dic_death.setdefault(0, []).append(i)
            discharge_index = np.where(kg.reg_ar[:, 45] == i)[0]
            for j in discharge_index:
                if kg.reg_ar[j][12] == kg.reg_ar[j][12]:
                    discharge_time_ = kg.reg_ar[j][12]
                    kg.dic_patient[i]['discharge_time'] = discharge_time_
            discharge_time = discharge_time_.split(' ')
            discharge_date = [np.int(l) for l in discharge_time[0].split('-')]
            discharge_date_value = (discharge_date[0] * 365.0 + discharge_date[1] * 30 + discharge_date[2]) * 24 * 60
            dischar_time_ = [np.int(l) for l in discharge_time[1].split(':')[0:-1]]
            discharge_time_value = dischar_time_[0] * 60.0 + dischar_time_[1]
            total_discharge_time_value = discharge_date_value + discharge_time_value
            kg.dic_patient[i]['discharge_value'] = total_discharge_time_value
            kg.dic_patient[i]['discharge_hour'] = np.int(
                np.floor((total_discharge_time_value - kg.dic_patient[i]['total_in_time_value']) / 60))

        if kg.dic_patient[i]['death_flag'] == 1:
            kg.dic_death.setdefault(1, []).append(i)
            death_index = np.where(kg.reg_ar[:, 45] == i)[0]
            for k in death_index:
                if kg.reg_ar[k][11] == kg.reg_ar[k][11]:
                    death_time_ = kg.reg_ar[k][11]
                    kg.dic_patient[i]['death_time'] = death_time_
            death_time = death_time_.split(' ')
            death_date = [np.int(l) for l in death_time[0].split('-')]
            death_date_value = (death_date[0] * 365.0 + death_date[1] * 30 + death_date[2]) * 24 * 60
            dead_time_ = [np.int(l) for l in death_time[1].split(':')[0:-1]]
            dead_time_value = dead_time_[0] * 60.0 + dead_time_[1]
            total_dead_time_value = death_date_value + dead_time_value
            kg.dic_patient[i]['death_value'] = total_dead_time_value
            kg.dic_patient[i]['death_hour'] = np.int(
                np.floor((total_dead_time_value - kg.dic_patient[i]['total_in_time_value']) / 60))

    age_total = []
    for i in kg.dic_demographic.keys():
        age = kg.dic_demographic[i]['Age']
        if age == 0:
            continue
        else:
            age_total.append(age)
    kg.age_mean = np.mean(age_total)
    kg.age_std = np.std(age_total)
    """
    com_file = '/home/tingyi.wanyan/comorbidity_matrix_20200710.csv'
    com = pd.read_csv(com_file)
    com_ar_rough = np.concatenate(np.array(com))
    com_ = []
    [com_.append(i.split(' ')) for i in com_ar_rough]
    com_ar  = np.array(com_)
    kg.com_ar = com_ar
    kg.com = com
    remove_symbol = np.array([i.replace('"','') for i in kg.com_ar[:,1]])
    kg.com_ar[:,1] = remove_symbol

    com_mapping_file = '/datadrive/user_tingyi.wanyan/RemappedMRNs.csv'
    com_mapping = pd.read_csv(com_mapping_file)
    com_mapping_ar = np.array(com_mapping)

    kg.com_mapping_ar = com_mapping_ar
    """
    """
    process_data = kg_process_data(kg)
    process_data.separate_train_test()
    LSTM_ = LSTM_model(kg, process_data)
    # LSTM_.config_model()
    # LSTM_.train()
    dhgm = dynamic_hgm(kg, process_data)
    feature = list(kg.dic_vital.keys()) + list(kg.dic_lab.keys())
    mlp = MLP_model(kg, process_data)
    """
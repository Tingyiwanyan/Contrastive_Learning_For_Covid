import numpy as np
import random
import math
import time
import pandas as pd
import json
from LSTM_icu import LSTM_model
from Data_process import kg_process_data
from Dynamic_hgm_icu_whole import dynamic_hgm
from MLP import MLP_model


class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """

    def __init__(self):
        file_path = '/datadrive/tingyi_wanyan/Registry_2020-10-15'
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
        self.dic_filter_patient = {}
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

        """
        get all patient with admit time
        """
        admit_time = np.where(self.reg_ar[:,1]==self.reg_ar[:,1])[0]
        self.admit = self.reg_ar[admit_time,:]
        covid_obv = np.where(self.admit[:,8]==self.admit[:,8])[0]
        self.covid_ar = self.admit[covid_obv,:]

        """
        filter out the first visit ID
        """
        for i in range(self.covid_ar.shape[0]):
            print("im here in filter visit ID")
            print(i)
            mrn_single = self.covid_ar[i,45]
            visit_id = self.covid_ar[i,65]
            if visit_id == visit_id:
                in_admit_time_single = self.covid_ar[i,1]
                if self.covid_ar[i,11] == self.covid_ar[i,11]:
                    death_flag = 1
                else:
                    death_flag = 0

                self.in_admit_time = in_admit_time_single.split(' ')
                in_admit_date = [np.int(i) for i in self.in_admit_time[0].split('-')]
                in_admit_date_value = (in_admit_date[0] * 365.0 + in_admit_date[1] * 30 + in_admit_date[2]) * 24 * 60
                self.in_admit_time_ = [np.int(i) for i in self.in_admit_time[1].split(':')[0:-1]]
                in_admit_time_value = self.in_admit_time_[0] * 60.0 + self.in_admit_time_[1]
                total_in_admit_time_value = in_admit_date_value + in_admit_time_value
                self.dic_filter_patient[mrn_single].setdefault(visit_id, []).append(self.in_admit_time_value)


if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.read_csv()
    kg.create_kg_dic()

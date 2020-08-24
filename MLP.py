import tensorflow as tf
import numpy as np
import random
import math
from itertools import groupby

class MLP_model():
    """
    Create MLP model for EHR data
    """
    def __init__(self,kg,data_process):
        """
        initializtion for varias variables
        """
        self.kg = kg
        self.data_process = data_process
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.batch_size = 16
        self.latent_dim = 100
        self.predict_winow_prior = 6
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))
        self.com_size = 12
        self.threshold = 0.5
        self.epoch = 1
        """
        define variables
        """
        self.input_y_logit = tf.keras.backend.placeholder(tf.float32,[None,2])
        self.input_x_vital = tf.keras.backend.placeholder(tf.float32,[None,self.item_size])
        self.input_x_lab = tf.keras.backend.placeholder(tf.float32,[None,self.lab_size])
        self.input_x_ = tf.concat([self.input_x_vital,self.input_x_lab],1)
        self.input_demo_ = tf.keras.backend.placeholder(tf.float32,[None,self.demo_size])
        self.input_x_com = tf.keras.backend.placeholder(tf.float32,[None,self.com_size])
        self.input_demo = tf.concat([self.input_demo_,self.input_x_com],1)
        self.input_x = tf.concat([self.input_x_,self.input_demo],1)

    def embed_layer(self):
        self.Dense_embed = tf.layers.dense(inputs=self.input_x,
                                            units=self.latent_dim,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            activation=tf.nn.relu)
    def softmax_loss(self):
        self.output_layer = tf.layers.dense(inputs=self.Dense_embed,
                                            units=2,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            activation=tf.nn.relu)

        self.logit_sig = tf.nn.softmax(self.output_layer)

        self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)
    
    def config_model(self):
        self.embed_layer()
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def assign_value_patient(self,patientid,start_time,end_time):
        self.one_sample = np.zeros(self.item_size)
        self.freq_sample = np.zeros(self.item_size)
        self.times = []
        for i in self.kg.dic_patient[patientid]['prior_time_vital'].keys():
            if (np.int(i) > start_time or np.int(i) ==start_time) and np.int(i) < end_time:
                self.times.append(i)
        
        for j in self.times:
            for i in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)].keys():
                mean = np.float(self.kg.dic_vital[i]['mean_value'])
                std = np.float(self.kg.dic_vital[i]['std'])
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)][i]])
                index = self.kg.dic_vital[i]['index']
                if std == 0:
                    self.one_sample[index] += 0
                    self.freq_sample[index] += 1
                else:
                    self.one_sample[index] += (np.float(ave_value) - mean) / std
                    self.freq_sample[index] += 1

        out_sample = self.one_sample/self.freq_sample
        for i in range(self.item_size):
            if math.isnan(out_sample[i]):
                out_sample[i] = 0

        return out_sample

    def assign_value_lab(self,patientid,start_time,end_time):
        self.one_sample_lab = np.zeros(self.lab_size)
        self.freq_sample_lab = np.zeros(self.lab_size)
        self.times_lab = []
        for i in self.kg.dic_patient[patientid]['prior_time_lab'].keys():
            if (np.int(i)>start_time or np.int(i)==start_time) and np.int(i)<end_time:
                self.times_lab.append(i)
        for j in self.times_lab:
            for i in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)].keys():
                mean = np.float(self.kg.dic_lab[i]['mean_value'])
                std = np.float(self.kg.dic_lab[i]['std'])
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i]])
                index = self.kg.dic_lab[i]['index']
                if std == 0:
                    self.one_sample_lab[index] += 0
                    self.freq_sample_lab[index] += 1
                else:
                    self.one_sample_lab[index] += (np.float(ave_value) - mean)/std
                    self.freq_sample_lab[index] += 1

        out_sample_lab = self.one_sample_lab/self.freq_sample_lab
        for i in range(self.lab_size):
            if math.isnan(out_sample_lab[i]):
                out_sample_lab[i] = 0

        return out_sample_lab

    def assign_value_demo(self,patientid):
        one_sample = np.zeros(self.demo_size)
        for i in self.kg.dic_demographic[patientid].keys():
            if i == 'race':
                race = self.kg.dic_demographic[patientid]['race']
                index = self.kg.dic_race[race]['index']
                one_sample[index] = 1
            elif i == 'Age':
                age = self.kg.dic_demographic[patientid]['Age']
                index = self.kg.dic_race['Age']['index']
                if age == 0:
                    one_sample[index]=age
                else:
                    one_sample[index] = (np.float(age)-self.kg.age_mean)/self.kg.age_std

            elif i == 'gender':
                gender = self.kg.dic_demographic[patientid]['gender']
                index = self.kg.dic_race[gender]['index']
                one_sample[index] = 1

        return one_sample
    
    def assign_value_com(self,patientid):
        one_sample = np.zeros(self.com_size)
        self.com_index = np.where(self.kg.com_mapping_ar[:,0] == patientid)[0][0]
        deidentify_index = self.kg.com_mapping_ar[self.com_index][1]
        self.map_index = np.where(deidentify_index == self.kg.com_ar[:,1])[0][0]
        one_sample[:] = [np.int(i) for i in self.kg.com_ar[self.map_index,4:]]

        return one_sample

    def get_batch_train_period(self,data_length,start_index,data):
        train_one_batch_vital = np.zeros((data_length,self.item_size))
        train_one_batch_lab = np.zeros((data_length,self.lab_size))
        train_one_batch_demo = np.zeros((data_length,self.demo_size))
        train_one_batch_com = np.zeros((data_length,self.com_size))
        one_batch_logit = np.zeros((data_length,2))
        for i in range(data_length):
            self.patient_id = data[start_index+i]
            flag = self.kg.dic_patient[self.patient_id]['death_flag']
            if flag == 0:
                start_time = self.kg.dic_patient[self.patient_id]['discharge_hour']-self.predict_winow_prior
                end_time = self.kg.dic_patient[self.patient_id]['discharge_hour']
            else:
                start_time = self.kg.dic_patient[self.patient_id]['death_hour']-self.predict_winow_prior
                end_time = self.kg.dic_patient[self.patient_id]['death_hour']

            self.one_data_vital = self.assign_value_patient(self.patient_id,start_time,end_time)
            self.one_data_lab = self.assign_value_lab(self.patient_id,start_time,end_time)
            train_one_batch_vital[i,:] = self.one_data_vital
            train_one_batch_lab[i,:] = self.one_data_lab
            self.one_data_demo = self.assign_value_demo(self.patient_id)
            self.one_data_com = self.assign_value_com(self.patient_id)
            train_one_batch_demo[i,:] = self.one_data_demo
            train_one_batch_com[i,:] = self.one_data_com
            if flag == 0:
                one_batch_logit[i,0] = 1
            else:
                one_batch_logit[i,1] = 1
        return train_one_batch_vital,train_one_batch_lab,one_batch_logit,train_one_batch_demo,train_one_batch_com

    def train(self):
        """
        train the system
        """
        iteration = np.int(np.floor(np.float(self.length_train)/self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch,self.train_one_batch_lab,self.logit_one_batch,self.train_one_batch_demo,self.train_one_batch_com = self.get_batch_train_period(self.batch_size,i*self.batch_size,self.train_data)
                self.err_ = self.sess.run([self.cross_entropy,self.train_step_cross_entropy],
                                        feed_dict={self.input_x_vital:self.train_one_batch,
                                                    self.input_x_lab:self.train_one_batch_lab,
                                                    self.input_y_logit:self.logit_one_batch,
                                                    self.input_demo_:self.train_one_batch_demo,
                                                    self.input_x_com:self.train_one_batch_com})
                print(self.err_[0])

    def test(self,data):
        test_length = len(data)
        test_data,self.test_data_lab,self.test_logit,self.test_demo,self.test_com = self.get_batch_train_period(test_length,0,data)
        self.logit_out = self.sess.run(self.logit_sig,feed_dict={self.input_x_vital:test_data,
                                                                self.input_demo_:self.test_demo,
                                                                self.input_x_lab:self.test_data_lab,
                                                                self.input_x_com:self.test_com})
        self.patient_embed = self.sess.run(self.Dense_embed,feed_dict={self.input_x_vital:test_data,
                                                                        self.input_demo_:self.test_demo,
                                                                        self.input_x_lab:self.test_data_lab,
                                                                        self.input_x_com:self.test_com})

        self.correct = 0
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0
        self.tp_correct = 0
        self.tp_neg = 0
        for i in range(test_length):
            if self.test_logit[i,1] == 1:
                self.tp_correct += 1
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] >self.threshold:
                print("im here")
                self.correct += 1
                self.tp_test += 1
                print(self.tp_test)
            if self.test_logit[i,1] == 0:
                self.tp_neg += 1
            if self.test_logit[i,1] == 1 and self.logit_out[i,1]<self.threshold:
                self.fn_test += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1]>self.threshold:
                self.fp_test += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1]<self.threshold:
                self.correct += 1


        self.precision_test = np.float(self.tp_test)/(self.tp_test+self.fp_test)
        self.recall_test = np.float(self.tp_test)/(self.tp_test+self.fn_test)

        self.f1_test = 2*(self.precision_test*self.recall_test)/(self.precision_test+self.recall_test)

        self.acc = np.float(self.correct)/test_length

        threshold = 0.0
        self.resolution = 0.05
        tp_test = 0
        fp_test = 0
        self.tp_total = []
        self.fp_total = []
        while(threshold<1.01):
            tp_test = 0
            fp_test = 0
            for i in range(test_length):
                if self.test_logit[i,1] == 1 and self.logit_out[i,1]>threshold:
                    tp_test += 1
                if self.test_logit[i,1] == 0 and self.logit_out[i,1]>threshold:
                    fp_test += 1

            tp_rate = tp_test/self.tp_correct
            fp_rate = fp_test/self.tp_neg
            self.tp_total.append(tp_rate)
            self.fp_total.append(fp_rate)
            threshold += self.resolution

    def cal_auc(self):
        self.area = 0
        self.tp_total.sort()
        self.fp_total.sort()
        for i in range(len(self.tp_total)-1):
            x = self.fp_total[i+1]-self.fp_total[i]
            y = (self.tp_total[i+1]+self.tp_total[i])/2
            self.area += x*y



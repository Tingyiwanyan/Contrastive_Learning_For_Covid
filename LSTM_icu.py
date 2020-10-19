import tensorflow as tf
import numpy as np
import random
import math
from itertools import groupby


class LSTM_model():
    """
    Create LSTM model for EHR data
    """

    def __init__(self, kg, data_process):
        """
        initialization for varies variables
        """
        self.kg = kg
        self.data_process = data_process
        # self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.batch_size = 16
        self.time_sequence = 4
        self.time_step_length = 6
        self.predict_window_prior = self.time_sequence * self.time_step_length
        self.latent_dim_cell_state = 100
        self.latent_dim_demo = 50
        self.epoch = 2
        self.train_time_window = self.time_sequence * self.time_step_length
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))
        self.latent_dim = self.item_size + self.lab_size
        self.com_size = 12
        self.input_seq = []
        self.threshold = 0.5
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.keras.backend.placeholder([None, self.latent_dim])
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])
        self.input_x_vital = tf.keras.backend.placeholder([None, self.time_sequence, self.item_size])
        self.input_x_lab = tf.keras.backend.placeholder([None, self.time_sequence, self.lab_size])
        self.input_x = tf.concat([self.input_x_vital, self.input_x_lab], 2)
        # self.input_y_diag_single = tf.placeholder(tf.float32,[None,self.diagnosis_size])
        # self.input_y_diag = tf.placeholder(tf.float32,[None,self.time_sequence,self.diagnosis_size])
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_softmax_convert = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(
                self.init_forget_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        # self.weight_softmax_convert = \
        #   tf.Variable(self.init_softmax_convert(shape=(self.latent_dim,self.diagnosis_size)))
        self.weight_output_gate = \
            tf.Variable(
                self.init_output_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        # self.bias_softmax_convert = tf.Variable(self.init_softmax_convert(shape=(self.diagnosis_size,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))
        """
        define input demographic and comorbidity features
        """
        self.input_demo_ = tf.keras.backend.placeholder([None, self.demo_size])
        self.input_x_com = tf.keras.backend.placeholder([None, self.com_size])
        self.input_demo = tf.concat([self.input_demo_, self.input_x_com], 1)

        """
        Define attention on Retain model for time
        """
        self.init_retain_b = tf.keras.initializers.he_normal(seed=None)
        self.init_retain_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_retain_w = tf.Variable(self.init_retain_weight(shape=(self.latent_dim, 1)))

        """
        Define attention on Retain model for feature variable
        """
        self.init_retain_variable_b = tf.keras.initializers.he_normal(seed=None)
        self.bias_retain_variable_b = tf.Variable(self.init_retain_variable_b(shape=(self.latent_dim,)))
        self.init_retain_variable_w = tf.keras.initializers.he_normal(seed=None)
        self.weight_retain_variable_w = tf.Variable(
            self.init_retain_variable_w(shape=(self.latent_dim, self.latent_dim)))

        """
        Define classification matrix for lstm
        """
        self.init_bias_classification_b = tf.keras.initializers.he_normal(seed=None)
        self.init_weight_classification_w = tf.keras.initializers.he_normal(seed=None)
        self.bias_classification_b = tf.Variable(self.init_bias_classification_b(shape=(1,)))
        self.weight_classification_w = tf.Variable(
            self.init_weight_classification_w(shape=(self.latent_dim + self.latent_dim_demo, 1)))

        """
        Define input projection
        """
        self.init_projection_b = tf.keras.initializers.he_normal(seed=None)
        self.bias_projection_b = tf.Variable(self.init_projection_b(shape=(self.latent_dim,)))
        self.init_projection_w = tf.keras.initializers.he_normal(seed=None)
        self.weight_projection_w = tf.Variable(
            self.init_projection_w(shape=(self.lab_size + self.item_size, self.latent_dim)))

    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        self.project_input = tf.math.add(tf.matmul(self.input_x, self.weight_projection_w), self.bias_projection_b)
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.project_input, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate, x_input_cur], 1)
            else:
                concat_cur = tf.concat([hidden_rep[i - 1], x_input_cur], 1)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_forget_gate), self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_info_gate), self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur, self.weight_cell_state), self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i == 0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state, info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur, self.weight_output_gate), self.bias_output_gate))
            hidden_current = tf.multiply(output_gate, cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)

        self.hidden_last = hidden_rep[self.time_sequence - 1]
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i], 1)
        self.hidden_rep = tf.concat(hidden_rep, 1)
        self.check = concat_cur

    def demo_layer(self):
        self.Dense_demo = tf.compat.v1.layers.dense(inputs=self.input_demo_,
                                                    units=self.latent_dim_demo,
                                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                    activation=tf.nn.relu)

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        #self.hidden_last_comb = tf.concat([self.hidden_last, self.Dense_demo], 1)

        self.hidden_att_e = tf.matmul(self.hidden_rep, self.weight_retain_w)
        self.hidden_att_e_softmax = tf.nn.softmax(self.hidden_att_e, 1)
        self.hidden_att_e_broad = tf.broadcast_to(self.hidden_att_e_softmax, [tf.shape(self.input_x_vital)[0],
                                                                              self.time_sequence,
                                                                              self.latent_dim])
        self.hidden_att_e_variable = tf.math.sigmoid(
            tf.math.add(tf.matmul(self.hidden_rep, self.weight_retain_variable_w), self.bias_retain_variable_b))
        # self.hidden_att_e_softmax = tf.nn.softmax(self.hidden_att_e, -1)
        self.parameter_mul = tf.multiply(self.hidden_att_e_broad, self.hidden_att_e_variable)
        self.hidden_mul_variable = tf.multiply(self.parameter_mul, self.project_input)
        # self.hidden_final = tf.reduce_sum(self.hidden_mul, 1)
        self.hidden_final = tf.reduce_sum(self.hidden_mul_variable, 1)
        self.hidden_last_comb = tf.concat([self.hidden_final, self.Dense_demo], 1)


        self.output_layer = tf.math.sigmoid(
            tf.math.add(tf.matmul(self.hidden_last_comb, self.weight_classification_w), self.bias_classification_b))
        # self.logit_sig = tf.math.sigmoid(self.output_layer)
        # self.logit_sig = tf.nn.softmax(self.output_layer)
        # self.cross_entropy = tf.reduce_mean(tf.math.negative(
        #    tf.reduce_sum(tf.math.multiply(self.input_y_diag_single, tf.log(self.logit_softmax)), reduction_indices=[1])))
        """
        self.cross_entropy = \
            tf.reduce_mean(
            tf.math.negative(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.math.multiply(
                            self.input_y_diag,tf.log(
                                self.logit_softmax)),reduction_indices=[1]),reduction_indices=[1])))
        """
        """
        self.cross_entropy = tf.reduce_mean(tf.math.negative(
            tf.reduce_sum(tf.math.multiply(self.input_y_logit, tf.log(self.logit_sig)), axis=1)),
            axis=0)
        """

        # self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        # self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)

        a = tf.constant((float(1)),shape=[self.batch_size,1])

        self.cross_entropy = tf.math.negative(tf.reduce_sum(tf.math.multiply(self.input_y_logit,tf.log(self.output_layer)),axis=0))+\
                             tf.math.negative(tf.reduce_sum(tf.math.multiply((a-self.input_y_logit),tf.log(a-self.output_layer)),axis=0))


        #self.bce = tf.keras.losses.BinaryCrossentropy()
        #self.cross_entropy = self.bce(self.input_y_logit, self.output_layer)

        #self.cross_entropy = tf.compat.v1.losses.hinge_loss(
            #self.input_y_logit, self.output_layer, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES)

        """
        Get interpretation matrix
        """

        self.braod_weight_variable = tf.broadcast_to(self.weight_projection_w, [tf.shape(self.input_x_vital)[0],
                                                                                self.time_sequence,
                                                                                self.latent_dim, self.latent_dim])

        self.exp_hidden_att_e_variable = tf.expand_dims(self.hidden_att_e_variable, axis=3)
        self.broad_hidden_att_e_variable = tf.broadcast_to(self.exp_hidden_att_e_variable,
                                                           [tf.shape(self.input_x_vital)[0],
                                                            self.time_sequence,
                                                            self.latent_dim, self.latent_dim])

        self.exp_hidden_att_e_broad = tf.expand_dims(self.hidden_att_e_broad, axis=3)
        self.broad_hidden_att_e = tf.broadcast_to(self.exp_hidden_att_e_broad, [tf.shape(self.input_x_vital)[0],
                                                                                self.time_sequence,
                                                                                self.latent_dim, self.latent_dim])
        self.project_weight_variable = tf.multiply(self.broad_hidden_att_e_variable, self.braod_weight_variable)
        self.project_weight_variable_final = tf.multiply(self.broad_hidden_att_e, self.project_weight_variable)

        """
        Get score important
        """

        self.time_feature_index = tf.constant([i for i in range(self.lab_size + self.item_size)])
        self.mortality_hidden_rep = tf.gather(self.weight_classification_w, self.time_feature_index, axis=0)
        #self.score_attention_ = tf.matmul(self.project_weight_variable_final,
        #                                  tf.expand_dims(tf.squeeze(self.mortality_hidden_rep), 1))

        self.score_attention_ = tf.matmul(self.project_weight_variable_final,self.mortality_hidden_rep)
        self.score_attention = tf.squeeze(self.score_attention_, [3])
        self.input_importance = tf.multiply(self.score_attention, self.input_x)


    def config_model(self):
        """
        Model configuration
        """
        self.lstm_cell()
        self.demo_layer()
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def assign_value_patient(self, patientid, start_time, end_time):
        self.one_sample = np.zeros(self.item_size)
        self.freq_sample = np.zeros(self.item_size)
        self.times = []
        for i in self.kg.dic_patient[patientid]['prior_time_vital'].keys():
            if (np.int(i) > start_time or np.int(i) == start_time) and np.int(i) < end_time:
                self.times.append(i)

        for j in self.times:
            for i in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)].keys():
                mean = np.float(self.kg.dic_vital[i]['mean_value'])
                std = np.float(self.kg.dic_vital[i]['std'])
                ave_value = np.mean(
                    [np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)][i]])
                index = self.kg.dic_vital[i]['index']
                if std == 0:
                    self.one_sample[index] += 0
                    self.freq_sample[index] += 1
                else:
                    self.one_sample[index] += (np.float(ave_value) - mean) / std
                    self.freq_sample[index] += 1

        out_sample = self.one_sample / self.freq_sample
        for i in range(self.item_size):
            if math.isnan(out_sample[i]):
                out_sample[i] = 0

        return out_sample

    def assign_value_lab(self, patientid, start_time, end_time):
        self.one_sample_lab = np.zeros(self.lab_size)
        self.freq_sample_lab = np.zeros(self.lab_size)
        self.times_lab = []
        for i in self.kg.dic_patient[patientid]['prior_time_lab'].keys():
            if (np.int(i) > start_time or np.int(i) == start_time) and np.int(i) < end_time:
                self.times_lab.append(i)
        for j in self.times_lab:
            for i in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)].keys():
                if i[-1] == 'A':
                    continue
                if i == "EOSINO":
                    continue
                if i == "EOSINO_PERC":
                    continue
                if i == "BASOPHIL":
                    continue
                if i == "BASOPHIL_PERC":
                    continue
                mean = np.float(self.kg.dic_lab[i]['mean_value'])
                std = np.float(self.kg.dic_lab[i]['std'])
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i]])
                index = self.kg.dic_lab[i]['index']
                if std == 0:
                    self.one_sample_lab[index] += 0
                    self.freq_sample_lab[index] += 1
                else:
                    self.one_sample_lab[index] += (np.float(ave_value) - mean) / std
                    self.freq_sample_lab[index] += 1

        out_sample_lab = self.one_sample_lab / self.freq_sample_lab
        for i in range(self.lab_size):
            if math.isnan(out_sample_lab[i]):
                out_sample_lab[i] = 0

        return out_sample_lab

    def assign_value_demo(self, patientid):
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
                    one_sample[index] = age
                else:
                    one_sample[index] = (np.float(age) - self.kg.age_mean) / self.kg.age_std
            elif i == 'gender':
                gender = self.kg.dic_demographic[patientid]['gender']
                index = self.kg.dic_race[gender]['index']
                one_sample[index] = 1
        return one_sample

    def assign_value_com(self, patientid):
        one_sample = np.zeros(self.com_size)
        self.com_index = np.where(self.kg.com_mapping_ar[:, 0] == patientid)[0][0]
        deindentify_index = self.kg.com_mapping_ar[self.com_index][1]
        self.map_index = np.where(deindentify_index == self.kg.com_ar[:, 1])[0][0]
        # com_index = np.where(self.kg.com_ar[:,0] == patientid)[0]
        one_sample[:] = [np.int(i) for i in self.kg.com_ar[self.map_index, 4:]]

        return one_sample

    """
    def get_batch_train(self,data_length,start_index,data):

        train_one_batch = np.zeros((data_length,self.time_sequence,self.item_size))
        train_one_batch_demo = np.zeros((data_length,self.demo_size))
        train_one_batch_com = np.zeros((data_length,self.com_size))
        one_batch_logit = np.zeros((data_length,2))
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time_vital'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            for j in self.time_seq_int:
                if time_index == self.time_sequence:
                    break
                #self.time_index = np.int(j)
                self.one_data = self.assign_value_patient(self.patient_id,j)
                train_one_batch[i,time_index,:] = self.one_data
                flag = self.kg.dic_patient[self.patient_id]['death_flag']
                if flag == 0:
                    one_batch_logit[i, 0] = 1
                else:
                    #death_time_length = self.kg.dic_patient[i]['death_hour']-
                    one_batch_logit[i, 1] = 1
                time_index += 1

            self.one_data_demo = self.assign_value_demo(self.patient_id)
            train_one_batch_demo[i,:] = self.one_data_demo

        return train_one_batch, one_batch_logit, train_one_batch_demo
    """

    def get_batch_train_period(self, data_length, start_index, data):
        """
        get period train data
        """
        train_one_batch_vital = np.zeros((data_length, self.time_sequence, self.item_size))
        train_one_batch_lab = np.zeros((data_length, self.time_sequence, self.lab_size))
        train_one_batch_demo = np.zeros((data_length, self.demo_size))
        train_one_batch_com = np.zeros((data_length, self.com_size))
        one_batch_logit = np.zeros((data_length, 1))
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            flag = self.kg.dic_patient[self.patient_id]['icu_label']
            for j in range(self.time_sequence):
                # start_time = float(j)*self.time_step_length
                # end_time = start_time + self.time_step_length
                if flag == 0:
                    #start_time = self.kg.dic_patient[self.patient_id][
                    #                 'discharge_hour'] - self.predict_window_prior + float(j) * self.time_step_length
                    pick_icu_hour = self.kg.mean_icu_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                    start_time = pick_icu_hour - self.predict_window_prior + float(j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                    self.check_start_time = start_time
                else:
                    start_time = self.kg.dic_patient[self.patient_id]['in_icu_hour'] - self.predict_window_prior + float(
                        j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                    self.check_start_time = start_time

                self.one_data_vital = self.assign_value_patient(self.patient_id, start_time, end_time)
                self.one_data_lab = self.assign_value_lab(self.patient_id, start_time, end_time)
                train_one_batch_vital[i, j, :] = self.one_data_vital
                train_one_batch_lab[i, j, :] = self.one_data_lab
            # flag = self.kg.dic_patient[self.patient_id]['death_flag']
            if flag == 1:
                one_batch_logit[i, 0] = 1
            """
            else:
                #death_time_length = self.kg.dic_patient[self.patient_id]['death_hour']-self.train_time_window
                #if death_time_length < self.death_predict_window:
                one_batch_logit[i,1] = 1
            """
            # else:
            # one_batch_logit[i,0] = 1
            self.one_data_demo = self.assign_value_demo(self.patient_id)
            # self.one_data_com = self.assign_value_com(self.patient_id)
            train_one_batch_demo[i, :] = self.one_data_demo
            # train_one_batch_com[i,:] = self.one_data_com

        return train_one_batch_vital, train_one_batch_lab, one_batch_logit, train_one_batch_demo, train_one_batch_com

    def train(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch, self.train_one_batch_lab, self.logit_one_batch, self.train_one_batch_demo, self.train_one_batch_com = self.get_batch_train_period(
                    self.batch_size, i * self.batch_size, self.train_data)
                self.err_ = self.sess.run(
                    [self.cross_entropy, self.train_step_cross_entropy, self.init_hiddenstate, self.output_layer],
                    feed_dict={self.input_x_vital: self.train_one_batch,
                               self.input_x_lab: self.train_one_batch_lab,
                               self.input_y_logit: self.logit_one_batch,
                               self.init_hiddenstate: init_hidden_state,
                               self.input_demo_: self.train_one_batch_demo})
                # self.input_x_com:self.train_one_batch_com})
                print(self.err_[0])

    def test(self, data):
        """
        test the system, return the accuracy of the model
        """
        test_length = len(data)
        init_hidden_state = np.zeros((test_length, self.latent_dim))
        test_data, self.test_data_lab, self.test_logit, self.test_demo, self.test_com = self.get_batch_train_period(
            test_length, 0, data)
        self.logit_out = self.sess.run(self.output_layer, feed_dict={self.input_x_vital: test_data,
                                                                     self.input_demo_: self.test_demo,
                                                                     self.input_x_lab: self.test_data_lab,
                                                                     self.input_x_com: self.test_com,
                                                                     self.init_hiddenstate: init_hidden_state})

        self.test_att_score = self.sess.run([self.score_attention, self.input_importance],
                                            feed_dict={self.input_x_vital: test_data,
                                            self.input_demo_:self.test_demo,
                                            self.input_x_lab:self.test_data_lab,
                                            self.input_x_com:self.test_com,
                                            self.init_hiddenstate:init_hidden_state})

        self.correct = 0
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0
        self.tp_correct = 0
        self.tp_neg = 0
        """
        for i in range(test_length):
            if self.test_logit[i,1] == 1:
                self.tp_correct += 1
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] > self.threshold:
                print("im here")
                self.correct += 1
                self.tp_test += 1
                print(self.tp_test)
            if self.test_logit[i,1] == 0:
                self.tp_neg += 1
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] < self.threshold:
                self.fn_test += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] > self.threshold:
                self.fp_test += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] < self.threshold:
                self.correct += 1
        """
        self.correct_predict_icu = []
        for i in range(test_length):
            if self.test_logit[i, 0] == 1:
                self.tp_correct += 1
            if self.test_logit[i, 0] == 1 and self.logit_out[i, 0] > self.threshold:
                self.correct_predict_icu.append(i)
                self.correct += 1
                self.tp_test += 1
            if self.test_logit[i, 0] == 0:
                self.tp_neg += 1
            if self.test_logit[i, 0] == 1 and self.logit_out[i, 0] < self.threshold:
                self.fn_test += 1
            if self.test_logit[i, 0] == 0 and self.logit_out[i, 0] > self.threshold:
                self.fp_test += 1
            if self.test_logit[i, 0] == 0 and self.logit_out[i, 0] < self.threshold:
                self.correct += 1

        self.correct_predict_icu = np.array(self.correct_predict_icu)

        feature_len = self.item_size + self.lab_size

        self.test_data_scores = self.test_att_score[1][self.correct_predict_icu, :, :]
        self.ave_data_scores = np.zeros((self.time_sequence, feature_len))

        count = 0
        value = 0

        for j in range(self.time_sequence):
            for p in range(feature_len):
                for i in range(self.correct_predict_icu.shape[0]):
                    if self.test_data_scores[i, j, p] != 0:
                        count += 1
                        value += self.test_data_scores[i, j, p]
                if count == 0:
                    continue
                self.ave_data_scores[j, p] = float(value / count)
                count = 0
                value = 0

        """
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0
        for i in range(test_length):
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] > self.threshold:
                self.tp_test += 1
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] < self.threshold:
                self.fn_test += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] > self.threshold:
                self.fp_test += 1
        """
        self.precision_test = np.float(self.tp_test) / (self.tp_test + self.fp_test)
        self.recall_test = np.float(self.tp_test) / (self.tp_test + self.fn_test)

        self.f1_test = 2 * (self.precision_test * self.recall_test) / (self.precision_test + self.recall_test)

        self.acc = np.float(self.correct) / test_length

        threshold = 0.0
        self.resolution = 0.01
        tp_test = 0
        fp_test = 0
        self.tp_total = []
        self.fp_total = []
        self.precision_total = []
        self.recall_total = []
        while (threshold < 1.01):
            tp_test = 0
            fp_test = 0
            fn_test = 0
            precision_test = 0
            for i in range(test_length):
                if self.test_logit[i, 0] == 1 and self.logit_out[i, 0] > threshold:
                    tp_test += 1
                if self.test_logit[i, 0] == 0 and self.logit_out[i, 0] > threshold:
                    fp_test += 1
                if self.test_logit[i, 0] == 1 and self.logit_out[i, 0] < threshold:
                    fn_test += 1
            self.check_fp_test = fp_test
            print(self.check_fp_test)
            self.check_tp_test = tp_test
            print(self.check_tp_test)
            if (tp_test + fp_test) == 0:
                precision_test = 1
            else:
                precision_test = np.float(tp_test) / (tp_test + fp_test)
            recall_test = np.float(tp_test) / (tp_test + fn_test)
            tp_rate = tp_test / self.tp_correct
            fp_rate = fp_test / self.tp_neg
            self.tp_total.append(tp_rate)
            self.fp_total.append(fp_rate)
            self.precision_total.append(precision_test)
            self.recall_total.append(recall_test)
            threshold += self.resolution

    def cal_auc(self):
        self.area = 0
        self.tp_total.sort()
        self.fp_total.sort()
        for i in range(len(self.tp_total) - 1):
            x = self.fp_total[i + 1] - self.fp_total[i]
            y = (self.tp_total[i + 1] + self.tp_total[i]) / 2
            self.area += x * y
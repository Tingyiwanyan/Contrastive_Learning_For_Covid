import tensorflow as tf
import numpy as np
import random
import math
import copy
from itertools import groupby

class dynamic_hgm():
    """
    Create dynamic HGM model
    """
    def __init__(self,kg,data_process):
        #tf.compat.v1.disable_v2_behavior()
        #tf.compat.v1.disable_eager_execution()
        self.kg = kg
        self.data_process = data_process
        #self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.batch_size = 16
        self.time_sequence = 6
        self.time_step_length = 1
        self.predict_window_prior = self.time_sequence*self.time_step_length
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.latent_dim_att = 100
        self.latent_dim_demo = 50
        self.epoch = 12
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))
        self.com_size = 12
        self.input_seq = []
        self.threshold = 0.5
        self.positive_lab_size = 5
        self.negative_lab_size = 10
        self.positive_sample_size = self.positive_lab_size + 1
        self.negative_sample_size = self.negative_lab_size + 1
        self.neighbor_pick_skip = 5
        self.neighbor_pick_neg = 10
        self.neighbor_death = len(kg.dic_death[1])
        self.neighbor_discharge = len(kg.dic_death[0])
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.keras.backend.placeholder([None, 1+self.positive_lab_size+self.negative_lab_size,self.latent_dim])
        self.input_y_logit = tf.keras.backend.placeholder([None, 2])
        self.input_x_vital = tf.keras.backend.placeholder([None,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.item_size])
        self.input_x_lab = tf.keras.backend.placeholder([None,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.lab_size])
        self.input_x = tf.concat([self.input_x_vital,self.input_x_lab],3)
        self.input_x_demo = tf.keras.backend.placeholder([None,1+self.positive_lab_size+self.negative_lab_size,self.demo_size])
        self.input_x_com = tf.keras.backend.placeholder([None,1+self.positive_lab_size+self.negative_lab_size,self.com_size])
        #self.input_x_demo = tf.concat([self.input_x_demo_,self.input_x_com],2)
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
                tf.Variable(self.init_forget_gate(shape=(self.item_size+self.lab_size+self.latent_dim,self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size+self.lab_size+self.latent_dim,self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size+self.lab_size+self.latent_dim,self.latent_dim)))
        self.weight_output_gate = \
            tf.Variable(self.init_output_gate(shape=(self.item_size+self.lab_size+self.latent_dim,self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))

        """
        Define LSTM variables plus attention
        """
        self.init_hiddenstate_att = tf.keras.backend.placeholder([None,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.latent_dim])
        self.input_x_vital_att = tf.keras.backend.placeholder([None,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.item_size])
        self.input_x_lab_att = tf.keras.backend.placeholder([None,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.lab_size])
        self.input_x_att = tf.concat([self.input_x_vital_att,self.input_x_lab_att],3)
        self.input_x_demo_att = tf.keras.backend.placeholder([None,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.demo_size])

        """
        Define relation model
        """
        self.shape_relation = (self.latent_dim+self.latent_dim_demo,)
        self.init_mortality = tf.keras.initializers.he_normal(seed=None)
        self.init_lab = tf.keras.initializers.he_normal(seed=None)
        """
        Define parameters
        """
        self.mortality = tf.keras.backend.placeholder([None,2,2])
        self.init_weight_mortality = tf.keras.initializers.he_normal(seed=None)
        self.weight_mortality = \
            tf.Variable(self.init_weight_mortality(shape=(2,self.latent_dim+self.latent_dim_demo)))
        self.bias_mortality = tf.Variable(self.init_weight_mortality(shape=(self.latent_dim+self.latent_dim_demo,)))

        self.lab_test = \
            tf.keras.backend.placeholder([None,self.positive_lab_size+self.negative_lab_size,self.item_size])
        self.weight_lab = \
            tf.Variable(self.init_weight_mortality(shape=(self.item_size,self.latent_dim)))
        self.bias_lab = tf.Variable(self.init_weight_mortality(shape=(self.latent_dim,)))
        """
        relation type 
        """
        self.relation_mortality = tf.Variable(self.init_mortality(shape=self.shape_relation))
        self.relation_lab = tf.Variable(self.init_lab(shape=self.shape_relation))

        """
        Define attention mechanism
        """
        self.init_weight_att_W = tf.keras.initializers.he_normal(seed=None)
        self.init_weight_vec_a = tf.keras.initializers.he_normal(seed=None)
        self.weight_att_W = tf.Variable(self.init_weight_att_W(shape=(self.latent_dim+self.latent_dim_demo,self.latent_dim_att+self.latent_dim_demo)))
        self.weight_vec_a = tf.Variable(self.init_weight_vec_a(shape=(2*(self.latent_dim_att+self.latent_dim_demo),1)))

    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.input_x, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate,x_input_cur],2)
            else:
                concat_cur = tf.concat([hidden_rep[i-1],x_input_cur],2)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_forget_gate),self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_info_gate),self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur,self.weight_cell_state),self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i == 0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state,info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_output_gate),self.bias_output_gate))
            hidden_current = tf.multiply(output_gate,cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)

        self.hidden_last = hidden_rep[self.time_sequence-1]
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i],1)
        self.hidden_rep = tf.concat(hidden_rep,1)
        self.check = concat_cur

    def lstm_cell_att(self):
        """
        build att model
        """
        cell_state = []
        hidden_rep = []
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.input_x_att,i,axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate_att,x_input_cur],2)
            else:
                concat_cur = tf.concat([hidden_rep[i-1],x_input_cur],2)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_forget_gate),self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_info_gate),self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur,self.weight_cell_state),self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur,cellstate_cur)
            if not i == 0:
                forget_cell_state = tf.multiply(forget_cur,cell_state[i-1])
                cellstate_cur = tf.math.add(forget_cell_state,info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_output_gate),self.bias_output_gate))
            hidden_current = tf.multiply(output_gate,cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)
        self.hidden_last = hidden_rep[self.time_sequence-1]
                

    def demo_layer(self):
        self.Dense_demo = tf.compat.v1.layers.dense(inputs=self.input_x_demo,
                                            units=self.latent_dim_demo,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            activation=tf.nn.relu)

    def demo_layer_att(self):
        self.Dense_demo = tf.compat.v1.layers.dense(inputs=self.input_x_demo_att,
                                            units=self.latent_dim_demo,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            activation=tf.nn.relu)

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        idx_origin = tf.constant([0])
        self.hidden_last_comb = tf.concat([self.hidden_last,self.Dense_demo],2)
        self.patient_lstm = tf.gather(self.hidden_last_comb,idx_origin,axis=1)
        self.output_layer = tf.compat.v1.layers.dense(inputs=self.patient_lstm,
                                           units=2,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        #self.logit_sig = tf.math.sigmoid(self.output_layer)
        self.logit_sig = tf.nn.softmax(self.output_layer)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig,self.input_y_logit)
        #self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        #self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)

    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        #self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        self.hidden_last_comb = tf.concat([self.hidden_last,self.Dense_demo],2)
        self.Dense_patient = self.hidden_last_comb
        #self.Dense_patient = tf.expand_dims(self.hidden_rep,2)

        self.Dense_mortality_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.mortality,self.weight_mortality),self.bias_mortality))

        self.Dense_mortality = tf.math.subtract(self.Dense_mortality_,self.relation_mortality)
        """
        self.Dense_lab_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.lab_test,self.weight_lab),self.bias_lab))

        self.Dense_lab = tf.math.add(self.Dense_lab_,self.relation_lab)
        """

    def build_att_mortality(self):
        """
        build attention model for mortality node
        """
        self.att_skip_latent = tf.matmul(self.x_att_skip,self.weight_att_W)
        self.x_skip_center_brod = tf.broadcast_to(self.x_skip_mor,[self.batch_size,self.neighbor_pick_skip,self.latent_dim_att+self.latent_dim_demo])
        self.att_skip_center = tf.matmul(self.x_skip_center_brod,self.weight_att_W)
        self.concat_att_skip = tf.concat([self.att_skip_center,self.att_skip_latent],axis=2)


        self.att_neg_latent = tf.matmul(self.x_att_neg,self.weight_att_W)
        self.x_neg_center_brod = tf.broadcast_to(self.x_negative_mor,[self.batch_size,self.neighbor_pick_neg,self.latent_dim_att+self.latent_dim_demo])
        self.att_neg_center = tf.matmul(self.x_neg_center_brod,self.weight_att_W)
        self.concat_att_neg = tf.concat([self.att_neg_center,self.att_neg_latent],axis=2)

        """
        times the weight vector a
        """
        self.weight_att_skip_a = tf.matmul(self.concat_att_skip,self.weight_vec_a)
        self.weight_att_neg_a = tf.matmul(self.concat_att_neg,self.weight_vec_a)

        self.soft_max_att_skip = tf.broadcast_to(tf.nn.softmax(self.weight_att_skip_a,axis=1),[self.batch_size,self.neighbor_pick_skip,self.latent_dim_att+self.latent_dim_demo])
        self.soft_max_att_neg = tf.broadcast_to(tf.nn.softmax(self.weight_att_neg_a,axis=1),[self.batch_size,self.neighbor_pick_neg,self.latent_dim_att+self.latent_dim_demo])

        self.att_rep_skip_mor = tf.multiply(self.soft_max_att_skip,self.att_skip_latent)
        self.att_rep_neg_mor = tf.multiply(self.soft_max_att_neg,self.att_neg_latent)

        self.att_rep_skip_mor_sum = tf.reduce_sum(self.att_rep_skip_mor,1)
        self.att_rep_neg_mor_sum = tf.reduce_sum(self.att_rep_neg_mor,1)

        self.att_rep_skip_mor_final = tf.nn.relu(self.att_rep_skip_mor_sum)
        self.att_rep_neg_mor_final = tf.nn.relu(self.att_rep_neg_mor_sum)





    def get_latent_rep_hetero(self):
        """
        Prepare data for SGNS loss function
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient,idx_origin,axis=1)
        #self.x_origin = self.hidden_last

        idx_skip_mortality = tf.constant([0])
        self.x_skip_mor = tf.gather(self.Dense_mortality,idx_skip_mortality,axis=1)
        idx_neg_mortality = tf.constant([1])
        self.x_negative_mor = tf.gather(self.Dense_mortality,idx_neg_mortality,axis=1)

        """
        item_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_item = tf.gather(self.Dense_lab,item_idx_skip,axis=1)
        item_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_item = tf.gather(self.Dense_lab,item_idx_negative,axis=1)

        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=1)
        self.x_negative = tf.concat([self.x_negative,self.x_negative_item],axis=1)
        """
        patient_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient,patient_idx_skip,axis=1)
        patient_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient,patient_idx_negative,axis=1)

        self.x_skip = tf.concat([self.x_skip_mor,self.x_skip_patient],axis=1)
        self.x_negative = tf.concat([self.x_negative_mor,self.x_negative_patient],axis=1)

    def get_latent_rep_hetero_att(self):
        """
        Prepare data for att SGNS loss
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient,idx_origin,axis=1)

        idx_skip_mortality = tf.constant([0])
        self.x_skip_mor = tf.gather(self.Dense_mortality,idx_skip_mortality,axis=1)
        idx_neg_mortality = tf.constant([1])
        self.x_negative_mor = tf.gather(self.Dense_mortality,idx_neg_mortality,axis=1)

        patient_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient,patient_idx_skip,axis=1)
        patient_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient,patient_idx_negative,axis=1)

        att_idx_skip = tf.constant([i+self.positive_lab_size+self.negative_lab_size+1 for i in range(self.neighbor_pick_skip)])
        self.x_att_skip = tf.gather(self.Dense_patient,att_idx_skip,axis=1)
        att_idx_neg = tf.constant([i+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+1 for i in range(self.neighbor_pick_neg)])
        self.x_att_neg = tf.gather(self.Dense_patient,att_idx_neg,axis=1)

        #self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        #self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)

        self.build_att_mortality()

        self.x_skip = tf.concat([tf.expand_dims(self.att_rep_skip_mor_final,axis=1),self.x_skip_patient],axis=1)
        self.x_negative = tf.concat([tf.expand_dims(self.att_rep_neg_mor_final,axis=1),self.x_negative_patient],axis=1)




    def get_positive_patient(self,center_node_index):
        self.patient_pos_sample_vital = np.zeros((self.time_sequence,self.positive_lab_size+1, self.item_size))
        self.patient_pos_sample_lab = np.zeros((self.time_sequence,self.positive_lab_size+1,self.lab_size))
        self.patient_pos_sample_demo = np.zeros((self.positive_lab_size+1,self.demo_size))
        self.patient_pos_sample_com = np.zeros((self.positive_lab_size+1,self.com_size))
        if self.kg.dic_patient[center_node_index]['death_flag'] == 0:
            flag = 0
            neighbor_patient = self.kg.dic_death[0]
        else:
            flag = 1
            neighbor_patient = self.kg.dic_death[1]
        time_seq = self.kg.dic_patient[center_node_index]['prior_time_vital'].keys()
        time_seq_int = [np.int(k) for k in time_seq]
        time_seq_int.sort()
        #time_index = 0
       # for j in self.time_seq_int:
        for j in range(self.time_sequence):
            #if time_index == self.time_sequence:
            #    break
            if flag == 0:
                start_time = self.kg.dic_patient[center_node_index]['discharge_hour']-self.predict_window_prior+ float(j)*self.time_step_length
                end_time = start_time + self.time_step_length
            else:
                start_time = self.kg.dic_patient[center_node_index]['death_hour']-self.predict_window_prior+float(j)*self.time_step_length
                end_time = start_time + self.time_step_length
            one_data_vital = self.assign_value_patient(center_node_index, start_time,end_time)
            one_data_lab = self.assign_value_lab(center_node_index,start_time,end_time)
            #one_data_demo = self.assign_value_demo(center_node_index)
            self.patient_pos_sample_vital[j,0,:] = one_data_vital
            self.patient_pos_sample_lab[j,0,:] = one_data_lab
            #time_index += 1
        one_data_demo = self.assign_value_demo(center_node_index)
        #one_data_com = self.assign_value_com(center_node_index)
        self.patient_pos_sample_demo[0,:] = one_data_demo
        #self.patient_pos_sample_com[0,:] = one_data_com
        for i in range(self.positive_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            one_data_demo = self.assign_value_demo(patient_id)
            #one_data_com = self.assign_value_com(patient_id)
            self.patient_pos_sample_demo[i+1,:] = one_data_demo
            #self.patient_pos_sample_com[i+1,:] = one_data_com
            #time_index = 0
            #for j in time_seq_int:
            for j in range(self.time_sequence):
                #if time_index == self.time_sequence:
                 #   break
                #self.time_index = np.int(j)
                #start_time = float(j)*self.time_step_length
                #end_time = start_time + self.time_step_length
                if flag == 0:
                    start_time = self.kg.dic_patient[patient_id]['discharge_hour']-self.predict_window_prior+float(j)*self.time_step_length
                    end_time = start_time + self.time_step_length
                else:
                    start_time = self.kg.dic_patient[patient_id]['death_hour']-self.predict_window_prior+float(j)*self.time_step_length
                    end_time = start_time + self.time_step_length
                one_data_vital = self.assign_value_patient(patient_id,start_time,end_time)
                one_data_lab = self.assign_value_lab(patient_id,start_time,end_time)
                self.patient_pos_sample_vital[j,i+1,:] = one_data_vital
                self.patient_pos_sample_lab[j,i+1,:] = one_data_lab
                #time_index += 1

    def get_negative_patient(self,center_node_index):
        self.patient_neg_sample_vital = np.zeros((self.time_sequence,self.negative_lab_size,self.item_size))
        self.patient_neg_sample_lab = np.zeros((self.time_sequence,self.negative_lab_size,self.lab_size))
        self.patient_neg_sample_demo = np.zeros((self.negative_lab_size,self.demo_size))
        self.patient_neg_sample_com = np.zeros((self.negative_lab_size,self.com_size))
        if self.kg.dic_patient[center_node_index]['death_flag'] == 0:
            neighbor_patient = self.kg.dic_death[1]
            flag = 1
        else:
            neighbor_patient = self.kg.dic_death[0]
            flag = 0
        for i in range(self.negative_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            time_index = 0
            one_data_demo = self.assign_value_demo(patient_id)
            #one_data_com = self.assign_value_com(patient_id)
            self.patient_neg_sample_demo[i,:] = one_data_demo
            #self.patient_neg_sample_com[i,:] = one_data_com
            #for j in time_seq_int:
            for j in range(self.time_sequence):
                #if time_index == self.time_sequence:
                 #   break
                #self.time_index = np.int(j)
                #start_time = float(j)*self.time_step_length
                #end_time = start_time + self.time_step_length
                if flag == 0:
                    start_time = self.kg.dic_patient[patient_id]['discharge_hour']-self.predict_window_prior+float(j)*self.time_step_length
                    end_time = start_time + self.time_step_length
                else:
                    start_time = self.kg.dic_patient[patient_id]['death_hour']-self.predict_window_prior+float(j)*self.time_step_length
                    end_time = start_time + self.time_step_length
                one_data_vital = self.assign_value_patient(patient_id,start_time,end_time)
                one_data_lab = self.assign_value_lab(patient_id,start_time,end_time)
                self.patient_neg_sample_vital[j,i,:] = one_data_vital
                self.patient_neg_sample_lab[j,i,:] = one_data_lab
                #time_index += 1


    """
    def get_negative_sample_rep(self):
        self.item_neg_sample = np.zeros((self.negative_lab_size,self.item_size))
        index = 0
        for i in self.neg_nodes_item:
            one_sample_neg_item = self.assign_value_item(i)
            self.item_neg_sample[index,:] = one_sample_neg_item
            index += 1
    """


    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size, self.latent_dim+self.latent_dim_demo])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.positive_sample_size, self.latent_dim+self.latent_dim_demo])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))


    def config_model(self):
        self.lstm_cell()
        self.demo_layer()
        #self.softmax_loss()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()
        self.SGNN_loss()
        self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        #self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def config_model_att(self):
        self.lstm_cell_att()
        self.demo_layer_att()
        self.build_dhgm_model()
        self.get_latent_rep_hetero_att()
        #self.build_att_mortality()
        self.SGNN_loss()
        self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        # self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def assign_value_patient(self,patientid,start_time,end_time):
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
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)][i]])
                index = self.kg.dic_vital[i]['index']
                if std == 0:
                    self.one_sample[index] += 0
                    self.freq_sample[index] += 1
                else:
                    self.one_sample[index] = (np.float(ave_value) - mean) / std
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
                    self.one_sample_lab[index] += (np.float(ave_value)-mean)/std
                    self.freq_sample_lab[index] += 1

        out_sample_lab = self.one_sample_lab/self.freq_sample_lab
        for i in range(self.lab_size):
            if math.isnan(out_sample_lab[i]):
                out_sample_lab[i] = 0
        
        return out_sample_lab

    def assign_value_demo(self, patientid):
        one_sample = np.zeros(self.demo_size)
        for i in self.kg.dic_demographic[patientid]['race']:
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
                    one_sample[index] = (np.float(age) - self.kg.age_mean)/self.kg.age_std
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


    def get_batch_train(self,data_length,start_index,data):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros((data_length, self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_lab = np.zeros((data_length,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.lab_size))
        train_one_batch_demo = np.zeros((data_length,1+self.positive_lab_size+self.negative_lab_size,self.demo_size))
        train_one_batch_com = np.zeros((data_length,1+self.positive_lab_size+self.negative_lab_size,self.com_size))
        #train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length,2,2))
        one_batch_logit = np.zeros((data_length,2))
        self.real_logit = np.zeros(data_length)
        #self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        #self.item_pos_sample = np.zeros((self.positive_lab_size, self.item_size))
        index_batch = 0
        index_increase = 0
        #while index_batch < data_length:
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            #if self.kg.dic_patient[self.patient_id]['item_id'].keys() == {}:
             #   index_increase += 1
              #  continue
            #index_batch += 1
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time_vital'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            flag = self.kg.dic_patient[self.patient_id]['death_flag']
            """
            if flag == 0:
                one_batch_logit[i,0,0] = 1
                one_batch_logit[i,1,1] = 1
            else:
                one_batch_logit[i,0,1] = 1
                one_batch_logit[i,1,0] = 1
                self.real_logit[i] = 1
            """
            if flag == 0:
                train_one_batch_mortality[i,0,:] = [1,0]
                train_one_batch_mortality[i,1,:] = [0,1]
                one_batch_logit[i, 0] = 1
            else:
                train_one_batch_mortality[i,0,:] = [0,1]
                train_one_batch_mortality[i,1,:] = [1,0]
                one_batch_logit[i, 1] = 1

            self.get_positive_patient(self.patient_id)
            self.get_negative_patient(self.patient_id)
            train_one_data_vital = np.concatenate((self.patient_pos_sample_vital,self.patient_neg_sample_vital),axis=1)
            train_one_data_lab = np.concatenate((self.patient_pos_sample_lab,self.patient_neg_sample_lab),axis=1)
            train_one_data_demo = np.concatenate((self.patient_pos_sample_demo,self.patient_neg_sample_demo),axis=0)
            train_one_data_com = np.concatenate((self.patient_pos_sample_com,self.patient_neg_sample_com),axis=0)
            train_one_batch_vital[i,:,:,:] = train_one_data_vital
            train_one_batch_lab[i,:,:,:] = train_one_data_lab
            train_one_batch_demo[i,:,:] = train_one_data_demo
            train_one_batch_com[i,:,:] = train_one_data_com

        return train_one_batch_vital,train_one_batch_lab,train_one_batch_demo,one_batch_logit, train_one_batch_mortality,train_one_batch_com


    def get_batch_train_att(self,data_length,start_index,data):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros((data_length, self.time_sequence,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.item_size))
        train_one_batch_lab = np.zeros((data_length,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.lab_size))
        train_one_batch_demo = np.zeros((data_length,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.demo_size))
        train_one_batch_com = np.zeros((data_length,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.com_size))
        #train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length,2,2))
        one_batch_logit = np.zeros((data_length,2))
        self.real_logit = np.zeros(data_length)
        #self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        #self.item_pos_sample = np.zeros((self.positive_lab_size, self.item_size))
        index_batch = 0
        index_increase = 0
        #while index_batch < data_length:
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            #if self.kg.dic_patient[self.patient_id]['item_id'].keys() == {}:
             #   index_increase += 1
              #  continue
            #index_batch += 1
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time_vital'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            flag = self.kg.dic_patient[self.patient_id]['death_flag']
            """
            if flag == 0:
                one_batch_logit[i,0,0] = 1
                one_batch_logit[i,1,1] = 1
            else:
                one_batch_logit[i,0,1] = 1
                one_batch_logit[i,1,0] = 1
                self.real_logit[i] = 1
            """
            if flag == 0:
                train_one_batch_mortality[i,0,:] = [1,0]
                train_one_batch_mortality[i,1,:] = [0,1]
                one_batch_logit[i, 0] = 1
            else:
                train_one_batch_mortality[i,0,:] = [0,1]
                train_one_batch_mortality[i,1,:] = [1,0]
                one_batch_logit[i, 1] = 1

            self.get_positive_patient(self.patient_id)
            self.get_negative_patient(self.patient_id)
            train_one_data_vital = np.concatenate((self.patient_pos_sample_vital,self.patient_neg_sample_vital),axis=1)
            train_one_data_vital = np.concatenate((train_one_data_vital,self.patient_pos_sample_vital[:,1:,:]),axis=1)
            train_one_data_vital = np.concatenate((train_one_data_vital,self.patient_neg_sample_vital),axis=1)
            train_one_data_lab = np.concatenate((self.patient_pos_sample_lab,self.patient_neg_sample_lab),axis=1)
            train_one_data_lab = np.concatenate((train_one_data_lab,self.patient_pos_sample_lab[:,1:,:]),axis=1)
            train_one_data_lab = np.concatenate((train_one_data_lab,self.patient_neg_sample_lab),axis=1)
            train_one_data_demo = np.concatenate((self.patient_pos_sample_demo,self.patient_neg_sample_demo),axis=0)
            train_one_data_demo = np.concatenate((train_one_data_demo,self.patient_pos_sample_demo[1:,:]),axis=0)
            train_one_data_demo = np.concatenate((train_one_data_demo,self.patient_neg_sample_demo),axis=0)
            train_one_data_com = np.concatenate((self.patient_pos_sample_com,self.patient_neg_sample_com),axis=0)
            train_one_batch_vital[i,:,:,:] = train_one_data_vital
            train_one_batch_lab[i,:,:,:] = train_one_data_lab
            train_one_batch_demo[i,:,:] = train_one_data_demo
            #train_one_batch_com[i,:,:] = train_one_data_com

        return train_one_batch_vital,train_one_batch_lab,train_one_batch_demo,one_batch_logit, train_one_batch_mortality,train_one_batch_com

    def get_batch_test(self,data_length,start_index,data):
        """
        get training batch data
        """
        train_one_batch = np.zeros((data_length, self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.item_size))
        #train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length,2,2))
        one_batch_logit = np.zeros((data_length,2))
        self.real_logit = np.zeros(data_length)
        #self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        #self.item_pos_sample = np.zeros((self.positive_lab_size, self.item_size))
        index_batch = 0
        index_increase = 0
        #while index_batch < data_length:
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            #if self.kg.dic_patient[self.patient_id]['item_id'].keys() == {}:
             #   index_increase += 1
              #  continue
            #index_batch += 1
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            flag = self.kg.dic_patient[self.patient_id]['flag']
            if flag == 0:
                train_one_batch_mortality[i,0,:] = [1,0]
                train_one_batch_mortality[i,1,:] = [0,1]
                one_batch_logit[i, 0] = 1
            else:
                train_one_batch_mortality[i,0,:] = [0,1]
                train_one_batch_mortality[i,1,:] = [1,0]
                one_batch_logit[i, 1] = 1

            self.get_positive_patient(self.patient_id)
            self.get_negative_patient(self.patient_id)
            train_one_data = np.concatenate((self.patient_pos_sample,self.patient_neg_sample),axis=1)
            train_one_batch[i,:,:,:] = train_one_data

        return train_one_batch,one_batch_logit, train_one_batch_mortality

    #def get_pos_neg_neighbor(self):
        

    def train(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size,1+self.positive_lab_size+self.negative_lab_size,self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train)/self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch_vital,self.train_one_batch_lab,self.train_one_batch_demo, self.one_batch_logit,self.one_batch_mortality,self.one_batch_com = self.get_batch_train(self.batch_size,i*self.batch_size,self.train_data)

                self.err_ = self.sess.run([self.negative_sum, self.train_step_neg],
                                     feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                self.input_x_lab: self.train_one_batch_lab,
                                                self.input_x_demo: self.train_one_batch_demo,
                                                #self.input_x_com: self.one_batch_com,
                                                #self.lab_test: self.one_batch_item,
                                                self.mortality: self.one_batch_mortality,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_[0])

                """
                self.err_lstm = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_sig],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_logit: self.one_batch_logit,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_lstm[0])
                """

    def train_att(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size,1+self.positive_lab_size+self.negative_lab_size+self.neighbor_pick_skip+self.neighbor_pick_neg,self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train)/self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch_vital,self.train_one_batch_lab,self.train_one_batch_demo, self.one_batch_logit,self.one_batch_mortality,self.one_batch_com = self.get_batch_train_att(self.batch_size,i*self.batch_size,self.train_data)

                self.err_ = self.sess.run([self.negative_sum, self.train_step_neg],
                                     feed_dict={self.input_x_vital_att: self.train_one_batch_vital,
                                                self.input_x_lab_att: self.train_one_batch_lab,
                                                self.input_x_demo_att: self.train_one_batch_demo,
                                                #self.input_x_com: self.one_batch_com,
                                                #self.lab_test: self.one_batch_item,
                                                self.mortality: self.one_batch_mortality,
                                                self.init_hiddenstate_att:init_hidden_state})
                print(self.err_[0])


    def test(self,data):
        test_length = len(data)
        init_hidden_state = np.zeros((test_length, 1+self.positive_lab_size+self.negative_lab_size, self.latent_dim))
        self.test_data_batch_vital,self.test_one_batch_lab,self.test_one_batch_demo,self.test_logit, self.test_mortality,self.test_com = self.get_batch_train(test_length, 0, data)
        self.test_patient = self.sess.run(self.Dense_patient, feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                                        self.input_x_lab: self.test_one_batch_lab,
                                                                        self.input_x_demo: self.test_one_batch_demo,
                                                                        #self.input_x_com: self.test_com,
                                                                        self.init_hiddenstate: init_hidden_state})[:,0,:]
        single_mortality = np.zeros((1,2,2))
        single_mortality[0][0][0] = 1
        single_mortality[0][1][1] = 1
        self.mortality_test = self.sess.run(self.Dense_mortality,feed_dict={self.mortality:single_mortality})[0]
        self.score = np.zeros(test_length)
        for i in range(test_length):
            embed_single_patient = self.test_patient[i]/np.linalg.norm(self.test_patient[i])
            embed_mortality = self.mortality_test[1]/np.linalg.norm(self.mortality_test[1])
            self.score[i] = np.matmul(embed_single_patient,embed_mortality.T)

        self.correct = 0
        self.tp_correct = 0
        self.tp_neg = 0
        for i in range(test_length):
            if self.test_logit[i,1] == 1:
                self.tp_correct += 1
            if self.test_logit[i,0] == 1:
                self.tp_neg += 1
            if self.score[i]<0 and self.test_logit[i,0] == 1:
                self.correct += 1
            if self.score[i]>0 and self.test_logit[i,1] == 1:
                self.correct += 1

        self.acc = np.float(self.correct)/test_length
        
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0
        for i in range(test_length):
            if self.score[i]>0 and self.test_logit[i,1] == 1:
                self.tp_test += 1
            if self.score[i]<0 and self.test_logit[i,1] == 1:
                self.fn_test += 1
            if self.score[i]>0 and self.test_logit[i,0] == 1:
                self.fp_test += 1
        
        self.precision_test = np.float(self.tp_test)/(self.tp_test+self.fp_test)
        self.recall_test = np.float(self.tp_test)/(self.tp_test+self.fn_test)
        self.f1_test = 2*(self.precision_test*self.recall_test)/(self.precision_test+self.recall_test)

        threshold = -1.01
        self.resolution = 0.05
        tp_test = 0
        fp_test = 0
        self.tp_total = []
        self.fp_total = []

        while(threshold<1.01):
            tp_test = 0
            fp_test = 0
            for i in range(test_length):
                if self.test_logit[i,1] == 1 and self.score[i]>threshold:
                    tp_test += 1
                if self.test_logit[i,0] == 1 and self.score[i]>threshold:
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


    def test_lstm(self):
        """
        test the system, return the accuracy of the model
        """
        init_hidden_state = np.zeros((self.length_test, self.latent_dim))
        self.test_data, self.test_logit,self.train_one_batch_item = self.get_batch_train(self.length_test,0,self.test_data)
        self.logit_out = self.sess.run(self.logit_sig,feed_dict={self.input_x: self.test_data,
                                            self.init_hiddenstate:init_hidden_state})
        self.correct = 0
        for i in range(self.length_test):
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] > self.threshold:
                self.correct += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] < self.threshold:
                self.correct += 1

        self.acc = np.float(self.correct)/self.length_test

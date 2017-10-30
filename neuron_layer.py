# coding: utf-8

import train
import plot_neurons_activity

import nest
import numpy as np
import matplotlib.pyplot as plt

import joblib

import random
import math
import sys


def _calc_mean_membrane_voltage(v, s, n):
    return v[np.where(s == n)].mean()

class NeuronLayer:

    def __init__(self, neuron_size, V_th = -55.0, tau_m = 10.0, neuron_model = "iaf_psc_alpha",
                 x_min = 0.0,
                 x_max = 0.0,
                 y_min = 0.0,
                 y_max = 0.0,
                 z_min = 0.0,
                 z_max = 0.0):

        self.neuron_model = neuron_model

        self.neurons = list(nest.Create(self.neuron_model, neuron_size,
                                        params = {"V_th": V_th, "tau_m": tau_m}))

        self.position = [(random.random() * (x_max - x_min) + x_min,
                          random.random() * (y_max - y_min) + y_min,
                          random.random() * (z_max - z_min) + z_min) for gid in self.neurons]
        
        self.detector = nest.Create("spike_detector", params = {"withgid": True, "withtime": True})
        self.meter = nest.Create("multimeter", params = {"withtime":True, "record_from":["V_m"]})

        # used only in readout layer. 
        self.presynaptic_neurons = [[] for n in self.neurons] # list of list
        self.conns = [[] for n in self.neurons] # list of list
        self.connected_liquid = None

        self.previous_delta_w = [np.zeros(1) for n in self.neurons]

        for i in range(neuron_size):
            nest.Connect([self.neurons[i]], self.detector)
            nest.Connect(self.meter, [self.neurons[i]])

    # 古いニューロンはどっかいく 消せるなら消したいけど...
    def replace_neurons(self, neuron_size, neuron_model):

        if len(self.neurons) > 0:
            nest.SetStatus(self.neurons, {"frozen": True})
            nest.SetStatus(self.detector, {"frozen": True})
        
        self.neuron_model = neuron_model

        self.neurons = nest.Create(neuron_model, neuron_size)
        self.detector = nest.Create("spike_detector", params = {"withgid": True, "withtime": True})
        self.meter = nest.Create("multimeter", params = {"withtime":True, "record_from":["V_m"]})

        self.presynaptic_neurons = [[] for n in self.neurons] # list of list
        self.conns = [[] for n in self.neurons] # list of list
        self.connected_liquid = None

        self.position = [None for n in self.neurons]

        for i in range(neuron_size):
            nest.Connect([self.neurons[i]], self.detector)
            nest.Connect(self.meter, [self.neurons[i]])


    def connect2liquid(self,
                       target_liquid_neurons,
                       connection_ratio,
                       inhibitory_connection_ratio,
                       weight_min,
                       weight_max,
                       delay_min,
                       delay_max):

        for ns in self.neurons:
            for nt in target_liquid_neurons.neurons:
                if random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(weight_min, weight_max) * sign
                    d = random.uniform(delay_min, delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})

    # connect at the probability of a * exp(-z_distance / b)
    def connect2liquid_prob_exp_z(self,
                                  target_liquid_neurons,
                                  a,
                                  b,
                                  inhibitory_connection_ratio,
                                  weight_min,
                                  weight_max,
                                  delay_min,
                                  delay_max):

        for ns in self.neurons:

            ns_pos = self.position[self.neurons.index(ns)]
            
            for nt in target_liquid_neurons.neurons:

                nt_pos = target_liquid_neurons.position[target_liquid_neurons.neurons.index(nt)]
                dist = abs(ns_pos[2] - nt_pos[2])
                
                if random.random() < a * math.exp(-dist / b):
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(weight_min, weight_max) * sign
                    d = random.uniform(delay_min, delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})


    # connect neurons at the probability of a * exp(-distance / b)
    def connect2liquid_prob_exp_dist(self,
                                     target_liquid_neurons,
                                     a,
                                     b,
                                     inhibitory_connection_ratio,
                                     weight_min,
                                     weight_max,
                                     delay_min,
                                     delay_max):

        for ns in self.neurons:

            ns_pos = self.position[self.neurons.index(ns)]
            
            for nt in target_liquid_neurons.neurons:
                
                nt_pos = target_liquid_neurons.position[target_liquid_neurons.neurons.index(nt)]
                dist = math.sqrt((ns_pos[0] - nt_pos[0])**2 + (ns_pos[1] - nt_pos[1])**2 + (ns_pos[2] - nt_pos[2])**2)
                
                if random.random() < a * math.exp(-dist / b):
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(weight_min, weight_max) * sign
                    d = random.uniform(delay_min, delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})
        
                

    # connect one_to_one with an uniform weight
    def connect2layer_one_to_one(self, target_neuron_layer, weight):

        nest.Connect(self.neurons, target_neuron_layer.neurons,
                     {"rule": "one_to_one"},
                     {"model": "static_synapse", "weight": weight})


    # st: spike timings (float list).
    def train_resume(self, st_ref):

        times_all = nest.GetStatus(self.connected_liquid.detector, keys = "events")[0]["times"]
        senders_all = nest.GetStatus(self.connected_liquid.detector, keys = "events")[0]["senders"]
        
        for i, n in enumerate(self.neurons):

            output_spike_train = self.get_spike_timings(i)
            desired_spike_train = st_ref

            size_pre = len(self.presynaptic_neurons[i])

            input_spike_train = []
            conn = []
            present_weight = []
            delta_w = None

            for ix in range(size_pre):
                sender_gid = self.connected_liquid.neurons[self.presynaptic_neurons[i][ix]]
                input_spike_train.append(times_all[np.where(senders_all == sender_gid)])
                                          
                conn.append(nest.GetConnections(source = [sender_gid],
                                                target = [self.neurons[i]]))
                present_weight.append(nest.GetStatus(conn[-1])[0]["weight"])

            delta_w = joblib.Parallel(n_jobs = -1)(joblib.delayed(train.resume)(input_spike_train[ix],
                                                                                 output_spike_train,
                                                                                 desired_spike_train,
                                                                                 a = 0.0025,
                                                                                 A_positive = 0.1 * 10**-10,
                                                                                 A_negative = -0.01*10**-10,
                                                                                 tau = 2.0) for ix in range(size_pre))

            for ix in range(size_pre):

                nest.SetStatus(conn[ix], {"weight": present_weight[ix] + delta_w[ix]})

                
    def train(self, tau_error, learning_ratio, momentum_learning_ratio, tolerance, filter_size):

        results = joblib.Parallel(n_jobs = -1)(joblib.delayed(train.train)(tau_error[ix], learning_ratio, momentum_learning_ratio, tolerance, filter_size, self.neurons[ix], np.array(self.presynaptic_neurons[ix], dtype = np.int32), self.conns[ix], self.previous_delta_w[ix], self.connected_liquid) for ix in range(len(self.neurons)))

        
        for ix in range(len(self.neurons)):
            if len(results[ix][0]) > 0:
                nest.SetStatus(self.conns[ix], [{"weight": nw} for nw in results[ix][0]])
            self.previous_delta_w[ix] = results[ix][1]
        
        # print np.linalg.norm([np.linalg.norm(l) for l in self.previous_delta_w])


    def set_input_current(self, current):
        
        nest.SetStatus(self.neurons, {"I_e": current})

    def get_spike_timings(self, ix):

        return self.get_detector_data(ix, "times")

    def get_meter_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.meter, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.meter, keys = "events")[0]["senders"]

        if neuron_ix is not None:
            return key_all[np.where(senders_all == self.neurons[neuron_ix])]
        else:
            return key_all
    
    def get_detector_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.detector, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.detector, keys = "events")[0]["senders"]

        if neuron_ix is not None:
            return key_all[np.where(senders_all == self.neurons[neuron_ix])]
        else:
            return key_all
    
    # filter_size [ms]
    def get_mean_membrane_voltage(self, filter_size):

        voltages = self.get_meter_data(None, "V_m")
        senders = self.get_meter_data(None, "senders")
        times = self.get_meter_data(None, "times")
        
        thresh = times[-1] - filter_size

        # 要らない過去のデータを捨てる
        voltages = voltages[np.where(times >= thresh)]
        senders = senders[np.where(times >= thresh)]

        result_list = np.zeros(len(self.neurons))
        
        for ix, n in enumerate(self.neurons):
            result_list[ix] = voltages[np.where(senders == n)].mean()

        # lambda式だとcan't pickle function objectsてエラーでた
        # result_list = joblib.Parallel(n_jobs = -1)(joblib.delayed(_calc_mean_membrane_voltage)(voltages, senders, n) for n in self.neurons)

        return np.array(result_list)

    # neuron_ix is the index of the self.neurons i.e. [0, len(self.neurons) - 1]
    def plot_spikes(self, neuron_ix, xticks = None, yticks = None, file_name = None, markersize = 0.5, marker = '_', grid = False):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return
        
        senders = self.get_detector_data(neuron_ix, "senders")
        times = self.get_detector_data(neuron_ix, "times")
        
        if len(times) == 0:
            sys.stderr.write("warning: no events recorded!\n")
            return 
            
        plt.figure()
        plt.plot(times, senders, marker, markersize = markersize)

        plt.xlabel("time [ms]")
        plt.ylabel("neuron ID")
        
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        
        if grid:
            plt.grid()
        
        if file_name is not None:
            plt.savefig(file_name)
            plt.close()
        else:
            plt.show()
        
        return

    def plot_V_m(self, neuron_ix, title = "", time_offset = 0.0,
                 xticks = None, yticks = None, file_name = None, grid = False):
        
        V_m = self.get_meter_data(neuron_ix, "V_m")
        times = self.get_meter_data(neuron_ix, "times") + time_offset
        plt.figure()
        plt.plot(times, V_m)

        plt.xlabel("time [ms]")
        plt.ylabel("membrane voltage [mV]")
        plt.title(title)

        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        
        if grid:
            plt.grid()
        
        if file_name is not None:
            plt.savefig(file_name)
            plt.close()
        else:
            plt.show()

        return 

    
    def raster_plot(self, title = "", file_name = None, **kwargs):
        
        plot_neurons_activity.from_device(self.detector, title = title, **kwargs)
        
        if file_name is not None:
            plt.savefig(file_name)
            plt.close()
        else:
            plt.show()

        return

    
    def num_of_spikes(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return

        senders = self.get_detector_data(neuron_ix, "senders")
        
        return len(senders)

    

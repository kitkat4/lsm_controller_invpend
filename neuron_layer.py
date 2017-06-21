# coding: utf-8

import resume

import nest
import numpy as np
import matplotlib.pyplot as plt

import random
import sys

class NeuronLayer:

    def __init__(self, neuron_size, V_th = -55.0, tau_m = 10.0):

        self.neurons = nest.Create("iaf_psc_alpha", neuron_size,
                                   params = {"V_th": V_th, "tau_m": tau_m})
        self.detectors = nest.Create("spike_detector", neuron_size,
                                     params = {"withgid": True, "withtime": True})
        self.meters = nest.Create("multimeter", neuron_size,
                                  params = {"withtime":True, "record_from":["V_m"]})

        # used only in readout layer. 
        self.presynaptic_neurons = {}
        self.presynaptic_neurons_detector = {}

        self.init_weight_min = 100.0
        self.init_weight_max = 1000.0
        self.delay_min = 0.5
        self.delay_max = 4.0
        
        for i in range(neuron_size):
            nest.Connect([self.neurons[i]], [self.detectors[i]])
            nest.Connect([self.meters[i]], [self.neurons[i]])
            

    def connect2liquid(self,
                       target_liquid_neurons,
                       connection_ratio,
                       inhibitory_connection_ratio):

        for ns in self.neurons:
            for nt in target_liquid_neurons.neurons:
                if random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(self.init_weight_min, self.init_weight_max) * sign
                    d = random.uniform(self.delay_min, self.delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})

    # connect one_to_one with an uniform weight
    def connect2layer_one_to_one(self, target_neuron_layer, weight):

        nest.Connect(self.neurons, target_neuron_layer.neurons,
                     {"rule": "one_to_one"},
                     {"model": "static_synapse", "weight": weight})

        
    # st: spike timings (float list).
    def train(self, st_ref):

        # if len(self.neurons) != 1:
        #     sys.stderr.write("error: get_spike_timings works only when one neuron is contained.\n")
        #     sys.exit()

        for i, n in enumerate(self.neurons):
        
            output_spike_train = self.get_spike_timings(i)
            desired_spike_train = st_ref
        
            for ix in range(len(self.presynaptic_neurons[n])):
                input_spike_train = nest.GetStatus([self.presynaptic_neurons_detector[n][ix]],
                                                   keys="events")[0]["times"]
                conn = nest.GetConnections([self.presynaptic_neurons[n][ix]],
                                           target = [self.neurons[i]])
                present_weight = nest.GetStatus(conn)[0]["weight"]
                delta_w =  resume.resume(input_spike_train,
                                         output_spike_train,
                                         desired_spike_train)
                nest.SetStatus(conn, {"weight": present_weight + delta_w})
        

    def set_input_current(self, current):
        
        nest.SetStatus(self.neurons, {"I_e": current})

    def get_spike_timings(self, ix):

        return nest.GetStatus(self.detectors, keys="events")[ix]["times"]
        
    # filter_size [ms]
    def get_mean_membrane_voltage(self, filter_size):

        meters_data = nest.GetStatus(self.meters, keys = "events")
        result_list = np.zeros(len(self.neurons))
        
        for ix in range(len(self.neurons)):
            voltages = meters_data[ix]["V_m"]
            times = meters_data[ix]["times"] # must be sorted

            if len(times) == 0:
                result_list[ix] = 0.0
                continue
        
            # 後ろのfilter_size[ms]の結果だけ取り出して平均を返せば良い
            thresh = times[-1] - filter_size
            size = 0
            for i in range(len(times)):
                if times[-i-1] >= thresh:
                    size += 1
                else:
                    break

            result_list[ix] = float(sum(voltages[-size:])) / size

        return result_list.mean()

    # neuron_ix is the index of the self.neurons i.e. [0, len(self.neurons) - 1]
    def plot(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return
        
        voltages = nest.GetStatus(self.meters, keys = "events")[neuron_ix]["V_m"]
        times = nest.GetStatus(self.meters, keys = "events")[neuron_ix]["times"]
        plt.figure()
        plt.plot(times, voltages)

        senders = nest.GetStatus(self.detectors, keys = "events")[neuron_ix]["senders"]
        times = nest.GetStatus(self.detectors, keys = "events")[neuron_ix]["times"]
        plt.figure()
        plt.plot(times, senders, '.')

        plt.show()
        
        return 

    

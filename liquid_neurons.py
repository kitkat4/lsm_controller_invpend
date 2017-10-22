# coding: utf-8

import plot_neurons_activity

import nest
# import nest.raster_plot

import matplotlib.pyplot as plt
import numpy as np

import random

class LiquidNeurons:
    
    def __init__(self,
                 neuron_size,
                 connection_ratio,
                 inhibitory_connection_ratio,
                 neuron_model,
                 weight_min,
                 weight_max,
                 delay_min,
                 delay_max):

        self.neuron_model = neuron_model

        self.neurons = nest.Create(self.neuron_model, neuron_size)

        self.detector = nest.Create("spike_detector", params = {"withgid": True, "withtime": True})
        self.meter = nest.Create("multimeter", params = {"withtime":True, "record_from":["V_m"]})


        
        for i in range(neuron_size): # connect one-to-one
            nest.Connect([self.neurons[i]], self.detector) 
            nest.Connect(self.meter, [self.neurons[i]])
            
        for ns in self.neurons:
            for nt in self.neurons:
                if ns == nt:    # no autapses 
                    pass
                elif random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(weight_min, weight_max) * sign
                    d = random.uniform(delay_min, delay_max)
                    
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})

    # 古いニューロンはどっかいく 消せるなら消したいけど...
    def replace_neurons(self, neuron_size, neuron_model):

        if len(self.neurons) > 0:
            nest.SetStatus(self.neurons, {"frozen": True})
            nest.SetStatus(self.detector, {"frozen": True})
        
        self.neuron_model = neuron_model

        self.neurons = nest.Create(neuron_model, neuron_size)
        self.detector = nest.Create("spike_detector", params = {"withgid": True, "withtime": True})
        self.meter = nest.Create("multimeter", params = {"withtime":True, "record_from":["V_m"]})


        for i in range(neuron_size):
            nest.Connect([self.neurons[i]], self.detector)
            nest.Connect(self.meter, [self.neurons[i]])

                    
    def connect(self,
                target_neuron_layer,
                connection_ratio,
                inhibitory_connection_ratio,
                weight_min,
                weight_max,
                delay_min,
                delay_max):

        target_neuron_layer.connected_liquid = self
        
        for ns in self.neurons:
            for nt in target_neuron_layer.neurons:
                if random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(weight_min, weight_max) * sign
                    d = random.uniform(delay_min, delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})
                    
                    # populate target_neuron_layer.presynaptic_neurons
                    s_ix = self.neurons.index(ns)
                    t_ix = target_neuron_layer.neurons.index(nt)
                    if s_ix not in target_neuron_layer.presynaptic_neurons[t_ix]:
                        target_neuron_layer.presynaptic_neurons[t_ix].append(s_ix)

    def get_meter_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.meter, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.meter, keys = "events")[0]["senders"]

        return key_all[np.where(senders_all == self.neurons[neuron_ix])]

    def get_detector_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.detector, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.detector, keys = "events")[0]["senders"]

        return key_all[np.where(senders_all == self.neurons[neuron_ix])]

                    

    # neuron_ix is the index of the self.neurons i.e. [0, len(self.neurons) - 1]
    def plot(self, neuron_ix, markersize = 2.5):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return
        
        times = self.get_meter_data(neuron_ix, "times")
        V_m = self.get_meter_data(neuron_ix, "V_m")
        plt.figure()
        plt.plot(times, V_m)

        senders = self.get_detector_data(neuron_ix, "senders")
        times = self.get_detector_data(neuron_ix, "times")
        plt.figure()
        plt.plot(times, senders, '.', markersize = markersize)

        plt.show()
        
        return

    
    def raster_plot(self, markersize = 2.5, title = "title", hist = True, **kwargs):
        
        plot_neurons_activity.from_device(self.detector, hist = hist, title = title, markersize = markersize, **kwargs)
        plt.show()
        return

    # 最新time_span[ms]でのスパイクの総数
    # time_span < 0なら記録に残ってるやつ全部
    def num_of_spikes(self, neuron_ix, time_span = -1.0):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return

        
        
            
        times = self.get_detector_data(neuron_ix, "times")
        if time_span < 0:
            return len(times)
        else:
            if len(times) == 0:
                return 0
            last = times[-1]
            return len(times[np.where(times >= last - time_span)])



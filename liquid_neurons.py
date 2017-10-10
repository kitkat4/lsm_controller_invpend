# coding: utf-8

import nest
import nest.raster_plot

import matplotlib.pyplot as plt

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
        self.meters = nest.Create("multimeter", neuron_size,
                                  params = {"withtime":True, "record_from":["V_m"]})


        
        for i in range(neuron_size): # connect one-to-one
            nest.Connect([self.neurons[i]], self.detector) 
            nest.Connect([self.meters[i]], [self.neurons[i]])
            
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
        self.meters = nest.Create("multimeter", neuron_size,
                                  params = {"withtime":True, "record_from":["V_m"]})


        for i in range(neuron_size):
            nest.Connect([self.neurons[i]], self.detector)
            nest.Connect([self.meters[i]], [self.neurons[i]])

                    
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
                    if nt not in target_neuron_layer.presynaptic_neurons:
                        target_neuron_layer.presynaptic_neurons[nt] = []
                    target_neuron_layer.presynaptic_neurons[nt].append(ns)
                    

    # neuron_ix is the index of the self.neurons i.e. [0, len(self.neurons) - 1]
    def plot(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return
        
        voltages = nest.GetStatus(self.meters, keys = "events")[neuron_ix]["V_m"]
        times = nest.GetStatus(self.meters, keys = "events")[neuron_ix]["times"]
        plt.figure()
        plt.plot(times, voltages)

        senders_all = nest.GetStatus(self.detector, keys = "events")[0]["senders"]
        times_all = nest.GetStatus(self.detector, keys = "events")[0]["times"]

        senders = []
        times = []
        
        for (itr_senders, itr_times) in zip(senders_all, times_all):
            if itr_senders == self.neurons[neuron_ix]:
                senders.append(itr_senders)
                times.append(itr_times)
        
        plt.figure()
        plt.plot(times, senders, '.')

        plt.show()
        
        return

    
    def raster_plot(self, title = "title", histogram = True):

        nest.raster_plot.from_device(self.detector, hist = histogram, title = title)
        plt.show()
        return

    def num_of_spikes(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return

        senders = nest.GetStatus(self.detector, keys = "events")[0]["senders"]
        times = nest.GetStatus(self.detector, keys = "events")[0]["times"]

        ret = 0
        for itr in senders:
            if itr == self.neurons[neuron_ix]:
                ret += 1

        return ret


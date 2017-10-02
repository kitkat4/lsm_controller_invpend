# coding: utf-8

import nest

import matplotlib.pyplot as plt

import random

class LiquidNeurons:
    
    def __init__(self,
                 neuron_size,
                 connection_ratio,
                 inhibitory_connection_ratio,
                 neuron_model = "iaf_psc_alpha"):

        self.neuron_model = neuron_model

        self.neurons = nest.Create(self.neuron_model, neuron_size)

        self.detectors = nest.Create("spike_detector", neuron_size,
                                     params = {"withgid": True, "withtime": True})
        self.meters = nest.Create("multimeter", neuron_size,
                                  params = {"withtime":True, "record_from":["V_m"]})


        self.init_weight_min = 100.0
        # self.init_weight_max = 1000.0
        self.init_weight_max = 500.0
        self.delay_min = 0.5
        self.delay_max = 4.0

        
        for i in range(neuron_size): # connect one-to-one
            nest.Connect([self.neurons[i]], [self.detectors[i]]) 
            nest.Connect([self.meters[i]], [self.neurons[i]])
            
        for ns in self.neurons:
            for nt in self.neurons:
                if ns == nt:    # no autapses 
                    pass
                elif random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(self.init_weight_min, self.init_weight_max) * sign
                    d = random.uniform(self.delay_min, self.delay_max)
                    
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})

    def connect(self,
                target_neuron_layer,
                connection_ratio,
                inhibitory_connection_ratio):

        for ns in self.neurons:
            for nt in target_neuron_layer.neurons:
                if random.random() < connection_ratio:
                    sign = -1 if random.random() < inhibitory_connection_ratio else 1
                    w = random.uniform(self.init_weight_min, self.init_weight_max) * sign
                    d = random.uniform(self.delay_min, self.delay_max)
                    nest.Connect([ns], [nt], {"rule": "one_to_one"},
                                 {"model": "static_synapse", "weight": w, "delay": d})

                    # populate target_neuron_layer.presynaptic_neurons
                    if nt not in target_neuron_layer.presynaptic_neurons:
                        target_neuron_layer.presynaptic_neurons[nt] = []
                        target_neuron_layer.presynaptic_neurons_detector[nt] = []
                    target_neuron_layer.presynaptic_neurons[nt].append(ns)
                    ns_detector = self.detectors[self.neurons.index(ns)]
                    target_neuron_layer.presynaptic_neurons_detector[nt].append(ns_detector)

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

    def num_of_spikes(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return

        return len(nest.GetStatus(self.detectors)[neuron_ix]["events"]["times"])


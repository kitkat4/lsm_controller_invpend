# coding: utf-8

import resume
import plot_neurons_activity

import nest
import numpy as np
import matplotlib.pyplot as plt

import joblib

import random
import sys

class NeuronLayer:

    def __init__(self, neuron_size, V_th = -55.0, tau_m = 10.0, neuron_model = "iaf_psc_alpha"):

        self.neuron_model = neuron_model

        self.neurons = nest.Create(self.neuron_model, neuron_size,
                                   params = {"V_th": V_th, "tau_m": tau_m})
        self.detector = nest.Create("spike_detector", params = {"withgid": True, "withtime": True})
        self.meter = nest.Create("multimeter", params = {"withtime":True, "record_from":["V_m"]})

        # used only in readout layer. 
        self.presynaptic_neurons = {}
        self.connected_liquid = None

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

    # connect one_to_one with an uniform weight
    def connect2layer_one_to_one(self, target_neuron_layer, weight):

        nest.Connect(self.neurons, target_neuron_layer.neurons,
                     {"rule": "one_to_one"},
                     {"model": "static_synapse", "weight": weight})


    # st: spike timings (float list).
    def train(self, st_ref):

        times_all = nest.GetStatus(self.connected_liquid.detector, keys = "events")[0]["times"]
        senders_all = nest.GetStatus(self.connected_liquid.detector, keys = "events")[0]["senders"]
        
        for i, n in enumerate(self.neurons):

            output_spike_train = self.get_spike_timings(i)
            desired_spike_train = st_ref

            size_pre = len(self.presynaptic_neurons[n])

            input_spike_train = []
            conn = []
            present_weight = []
            delta_w = None

            for ix in range(size_pre):
                input_spike_train.append(times_all[np.where(senders_all == self.presynaptic_neurons[n][ix])])
                                          
                conn.append(nest.GetConnections([self.presynaptic_neurons[n][ix]],
                                                target = [self.neurons[i]]))
                present_weight.append(nest.GetStatus(conn[-1])[0]["weight"])

            delta_w = joblib.Parallel(n_jobs = -1)(joblib.delayed(resume.resume)(input_spike_train[ix],
                                                                                 output_spike_train,
                                                                                 desired_spike_train) for ix in range(size_pre))

            for ix in range(size_pre):

                nest.SetStatus(conn[ix], {"weight": present_weight[ix] + delta_w[ix]})

    def set_input_current(self, current):
        
        nest.SetStatus(self.neurons, {"I_e": current})

    def get_spike_timings(self, ix):

        return self.get_detector_data(ix, "times")

    def get_meter_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.meter, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.meter, keys = "events")[0]["senders"]

        return key_all[np.where(senders_all == self.neurons[neuron_ix])]

    def get_detector_data(self, neuron_ix, key):
        
        key_all = nest.GetStatus(self.detector, keys = "events")[0][key]
        senders_all = nest.GetStatus(self.detector, keys = "events")[0]["senders"]

        return key_all[np.where(senders_all == self.neurons[neuron_ix])]

    
    # filter_size [ms]
    def get_mean_membrane_voltage(self, filter_size):

        result_list = np.zeros(len(self.neurons))
        
        for ix in range(len(self.neurons)):
            voltages = self.get_meter_data(ix, "V_m")
            times = self.get_meter_data(ix, "times")

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
    def plot(self, neuron_ix, markersize = 2.5):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return
        
        V_m = self.get_meter_data(neuron_ix, "V_m")
        times = self.get_meter_data(neuron_ix, "times")
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

    
    def num_of_spikes(self, neuron_ix):

        if neuron_ix not in range(len(self.neurons)):
            sys.stderr.write("warning: NeuronLayer.plot neuron_ix is out of range.\n")
            return

        senders = self.get_detector_data(neuron_ix, "senders")
        
        return len(senders)

    

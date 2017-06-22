# coding: utf-8

import neuron_layer
import liquid_neurons
import nest

import sys

class Lsm:

    def __init__(self,
                 input_neurons_theta_size,
                 input_neurons_theta_dot_size,
                 liquid_neurons_size,
                 readout_neurons_tau1_size,
                 readout_neurons_tau2_size,
                 output_layer_weight = 500.0):

        # create neurons
        self.input_layer_theta = neuron_layer.NeuronLayer(input_neurons_theta_size,
                                                           tau_m = float(10**100))
        self.input_layer_theta_dot = neuron_layer.NeuronLayer(input_neurons_theta_dot_size,
                                                              tau_m = float(10**100))
        
        self.liquid_neurons = liquid_neurons.LiquidNeurons(liquid_neurons_size, 0.3, 0.25)
        
        self.readout_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size)
        self.readout_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau2_size)

        self.output_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        self.output_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        
        # connect layers
        self.input_layer_theta.connect2liquid(self.liquid_neurons, 0.3, 0.25)
        self.input_layer_theta_dot.connect2liquid(self.liquid_neurons, 0.3, 0.25)
        self.liquid_neurons.connect(self.readout_layer_tau1, 0.3, 0.25)
        self.liquid_neurons.connect(self.readout_layer_tau2, 0.3, 0.25)
        self.readout_layer_tau1.connect2layer_one_to_one(self.output_layer_tau1,
                                                         weight = output_layer_weight)
        self.readout_layer_tau2.connect2layer_one_to_one(self.output_layer_tau2,
                                                         weight = output_layer_weight)

    # i_theta, i_theta_dot [pA]
    def simulate(self, sim_time, i_theta, i_theta_dot, filter_size):

        self.input_layer_theta.set_input_current(i_theta)
        self.input_layer_theta_dot.set_input_current(i_theta_dot)

        nest.Simulate(sim_time)

        ret1 = self.output_layer_tau1.get_mean_membrane_voltage(filter_size)
        ret2 = self.output_layer_tau2.get_mean_membrane_voltage(filter_size)
        
        return (ret1, ret2)

    # st: spike train (float list).
    # The dynamic state of the network will be reset.
    def train(self,
              i_theta,
              i_theta_dot,
              st_tau1_ref,
              st_tau2_ref,
              update_num = 100,
              sim_time = 1000.0,
              print_message = True):

        for i in range(update_num):

            if print_message:
                sys.stdout.write("training: " + str(i+1) + "/" + str(update_num) +  "    \r")
                sys.stdout.flush()
                
            nest.ResetNetwork()
            
            self.input_layer_theta.set_input_current(i_theta)
            self.input_layer_theta_dot.set_input_current(i_theta_dot)

            nest.Simulate(sim_time)
        
            self.readout_layer_tau1.train(st_tau1_ref)
            self.readout_layer_tau2.train(st_tau2_ref)

        if print_message:
            sys.stdout.write("\n")
            
        nest.ResetNetwork()        






    

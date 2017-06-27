# coding: utf-8

import lsm

import numpy as np
import nest

import math
import sys

class LsmController:

    def __init__(self,
                 input_neurons_theta_size,
                 input_neurons_theta_dot_size,
                 liquid_neurons_size,
                 readout_neurons_tau1_size,
                 readout_neurons_tau2_size,
                 output_layer_weight = 100.0,
                 filter_size = 3.0,
                 thread_num = 1):

        # Don't make any nest nodes or connections before this line!
        nest.SetKernelStatus({"local_num_threads": thread_num})

        nest.set_verbosity("M_ERROR") # suppress trivial messages
        sys.stdout.write("Initializing LsmController ... ")

        self.tau1 = 0.0
        self.tau2 = 0.0

        self.output_layer_weight = output_layer_weight
        self.filter_size = filter_size

        self.lsm = lsm.Lsm(input_neurons_theta_size,
                           input_neurons_theta_dot_size,
                           liquid_neurons_size,
                           readout_neurons_tau1_size,
                           readout_neurons_tau2_size,
                           output_layer_weight = self.output_layer_weight)

        sys.stdout.write("done.\n")

    def get_tau(self):

        return self.tau1 - self.tau2

    # theta [rad]
    def train(self, theta, theta_dot, tau1_ref, tau2_ref,
              update_num = 100, sim_time = 1000.0, print_message = True):

        #vec_conv_tau2spike_train = np.vectorize(self._conv_tau2spike_train)

        f_theta = self.vec_conv_theta2freq(theta)
        f_theta_dot = self.vec_conv_theta_dot2freq(theta_dot)
        i_theta = self.vec_conv_freq2current(f_theta)
        i_theta_dot = self.vec_conv_freq2current(f_theta_dot)

        st_tau1_ref = self.vec_conv_tau2spike_train(tau1_ref, sim_time)
        st_tau2_ref = self.vec_conv_tau2spike_train(tau2_ref, sim_time)

        self.lsm.train(i_theta,
                       i_theta_dot,
                       st_tau1_ref,
                       st_tau2_ref,
                       update_num = update_num,
                       sim_time = sim_time,
                       print_message = print_message)


    # sim_time [ms]
    def simulate(self, sim_time, theta, theta_dot):

        f_theta = self._conv_theta2freq(theta)
        f_theta_dot = self._conv_theta_dot2freq(theta_dot)
        i_theta = self._conv_freq2current(f_theta)
        i_theta_dot = self._conv_freq2current(f_theta_dot)

        (tau1_voltage, tau2_voltage) = self.lsm.simulate(sim_time,
                                                         i_theta,
                                                         i_theta_dot,
                                                         filter_size = self.filter_size)

        self._update_tau(tau1_voltage, tau2_voltage)


    # [-3 [rad], 3 [rad]] -> [100 [Hz], 400 [Hz]]
    def _conv_theta2freq(self, theta):

        return theta * 50.0 + 250.0

    def vec_conv_theta2freq(self, theta):
        vec_conv_theta2freq = np.vectorize(self._conv_theta2freq)
        return vec_conv_theta2freq(theta)


    # [-10 [rad/s], 10 [rad/s]] -> [100 [Hz], 400 [Hz]]
    def _conv_theta_dot2freq(self, theta_dot):

        return theta_dot * 15 + 250.0

    def vec_conv_theta_dot2freq(self, theta_dot):
        vec_conv_theta_dot2freq = np.vectorize(self._conv_theta_dot2freq)
        return vec_conv_theta_dot2freq(theta_dot)


    # [0 [Nm], 100 [Nm]] -> [-70 [mV], -50 [mV]]  許容する最大トルクはこれでいいのか？
    def _conv_tau2voltage(self, tau):

        return tau * 0.2 - 70.0

    def vec_conv_tau2voltage(self, tau):
        vec_conv_tau2voltage = np.vectorize(self._conv_tau2voltage)
        return vec_conv_tau2voltage(tau)

    # returns current[pA]
    def _conv_freq2current(self, freq): # tau_m が十分大きいニューロンを想定

        C_m = 250.0             # [pF]
        V_th = -55.0            # [mV]
        E_L = -70.0             # [mV]
        t_ref = 2.0             # [ms]
        return C_m * (V_th - E_L) * freq / (1000.0 - t_ref * freq)

    def vec_conv_freq2current(self, freq):
        vec_conv_freq2current = np.vectorize(self._conv_freq2current)
        return vec_conv_freq2current(freq)


    def _conv_tau2spike_train(self, tau, time_span):

        voltage = self._conv_tau2voltage(tau)
        weight = self.output_layer_weight
        tau_syn = 2.0
        C_m = 250.0
        tau_m = 10.0
        E_L = -70.0

        I = (C_m / tau_m) * (voltage - E_L) # [pA]
        charge_per_spike = weight * tau_syn * math.e / 1000.0 # [pC]
        freq = I / charge_per_spike
        return [round(x, 1) for x in np.linspace(1, time_span, freq)]

    def calculate_I(self, C_m, tau_m, voltage, E_L):
        return (C_m / tau_m) * (voltage - E_L)

    def vec_calculate_I(self, C_m, tau_m, voltage, E_L):
        vec_calculate_I = np.vectorize(self.calculate_I)
        return vec_calculate_I(C_m, tau_m, voltage, E_L)

    def freq(self, I, charge_per_spike, time_span):
        freq = I / charge_per_spike
        return [round(x, 1) for x in np.linspace(1, time_span, freq)]

    def vec_freq(self, I, charge_per_spike, time_span):
        freq = []
        for i in range(I.size):
            freq.append(self.freq(I[i], charge_per_spike, time_span))
        return freq

    def vec_conv_tau2spike_train(self, tau, time_span):

        voltage = self.vec_conv_tau2voltage(tau)
        weight = self.output_layer_weight
        tau_syn = 2.0
        C_m = 250.0
        tau_m = 10.0
        E_L = -70.0

        I = self.vec_calculate_I(C_m, tau_m, voltage, E_L) # [pA]
        charge_per_spike = weight * tau_syn * math.e / 1000.0 # [pC]
        freq = self.vec_freq(I, charge_per_spike, time_span)
        return freq


    # [-70 [mv], -50 [mv]] -> [0 [Nm], 100 [Nm]]
    def _update_tau(self, tau1_voltage, tau2_voltage):

        self.tau1 = tau1_voltage * 5.0 + 350.0
        self.tau2 = tau2_voltage * 5.0 + 350.0

# coding: utf-8

import lsm

import numpy as np
import nest

import math
import sys

import multiprocessing

# Don't call nest.Simulate outside this class.
# Otherwise LsmController.total_sim_time will get wrong and
# LsmController.train will fail to work properly.
class LsmController:

    def __init__(self,
                 input_neurons_theta_size,
                 input_neurons_theta_dot_size,
                 liquid_neurons_size,
                 readout_neurons_tau1_size,
                 readout_neurons_tau2_size,
                 output_layer_weight = 100.0,
                 thread_num = -1):
        
        
        # Don't make any nest nodes or connections before this line!
        nest.SetKernelStatus({"local_num_threads": thread_num if thread_num > 0 else multiprocessing.cpu_count()})
        
        nest.set_verbosity("M_ERROR") # suppress trivial messages

        self.total_sim_time = 0.0
        
        self.tau1 = np.zeros(readout_neurons_tau1_size)
        self.tau2 = np.zeros(readout_neurons_tau2_size)

        self.output_layer_weight = output_layer_weight
        
        self.lsm = lsm.Lsm(input_neurons_theta_size,
                           input_neurons_theta_dot_size,
                           liquid_neurons_size,
                           readout_neurons_tau1_size,
                           readout_neurons_tau2_size,
                           output_layer_weight = self.output_layer_weight)

    def get_tau(self):
        
        return self.tau1.mean() - self.tau2.mean()

    # theta [rad]
    # The dynamic state of the network will be reset.
    def train_resume(self, theta, theta_dot, tau1_ref, tau2_ref,
                     update_num = 100, sim_time = 1000.0, print_message = True):

        f_theta1 = self._conv_theta2freq(theta) if theta >= 0 else 0
        f_theta2 = self._conv_theta2freq(-theta) if theta < 0 else 0        
        f_theta_dot1 = self._conv_theta_dot2freq(theta_dot) if theta_dot >= 0 else 0
        f_theta_dot2 = self._conv_theta_dot2freq(-theta_dot) if theta_dot < 0 else 0
        i_theta1 = self._conv_freq2current(f_theta1)
        i_theta2 = self._conv_freq2current(f_theta2)
        i_theta_dot1 = self._conv_freq2current(f_theta_dot1)
        i_theta_dot2 = self._conv_freq2current(f_theta_dot2)
        

        st_tau1_ref = self._conv_tau2spike_train(tau1_ref, sim_time)
        st_tau2_ref = self._conv_tau2spike_train(tau2_ref, sim_time)

        self.lsm.train_resume(i_theta1,
                              i_theta2,
                              i_theta_dot1,
                              i_theta_dot2,
                              st_tau1_ref,
                              st_tau2_ref,
                              update_num = update_num,
                              sim_time = sim_time,
                              print_message = print_message)
        
        self.total_sim_time += sim_time * update_num

    # theta [rad]       
    # The dynamic state of the network will be reset.
    # 出力がtoleranceを超えて外れるreadout neuronへの接続の重みを，
    # sim_time間のその結合の平均周波数×learning_ratioだけ増減する
    def train(self, theta, theta_dot, tau1_ref, tau2_ref,
              learning_ratio, momentum_learning_ratio, tau1_tolerance, tau2_tolerance,
              sim_time, filter_size):

        nest.ResetNetwork()

        self.simulate(sim_time, theta, theta_dot, filter_size)

        tau1_error = self.tau1 - tau1_ref
        tau2_error = self.tau2 - tau2_ref

        self.lsm.train(tau1_error, tau2_error, learning_ratio, momentum_learning_ratio,
                       tau1_tolerance, tau2_tolerance, filter_size)

    
        nest.ResetNetwork()
        
        return 
        

    # sim_time [ms]
    # _update_tauで更新されるトルクは最後のfilter_size [ms]の平均となる
    def simulate(self, sim_time, theta, theta_dot, filter_size = 1.0):

        f_theta1 = self._conv_theta2freq(theta if theta >= 0 else 0.0)
        f_theta2 = self._conv_theta2freq(-theta if theta < 0 else 0.0)
        f_theta_dot1 = self._conv_theta_dot2freq(theta_dot if theta_dot >= 0 else 0.0)
        f_theta_dot2 = self._conv_theta_dot2freq(-theta_dot if theta_dot < 0 else 0.0)
        i_theta1 = self._conv_freq2current(f_theta1)
        i_theta2 = self._conv_freq2current(f_theta2)
        i_theta_dot1 = self._conv_freq2current(f_theta_dot1)
        i_theta_dot2 = self._conv_freq2current(f_theta_dot2)

        self.lsm.simulate(sim_time, i_theta1, i_theta2, i_theta_dot1, i_theta_dot2)

        (tau1_voltage, tau2_voltage) = self.lsm.get_mean_membrane_voltage(filter_size)
        
        self.total_sim_time += sim_time

        self._update_tau(tau1_voltage, tau2_voltage)

    def save(self, file_name):
        
        self.lsm.save(file_name)

    def load(self, file_name):

        self.lsm.load(file_name)
        self.tau1 = np.zeros(len(self.lsm.readout_layer_tau1.neurons))
        self.tau2 = np.zeros(len(self.lsm.readout_layer_tau2.neurons))
        
    
    # [0 [rad], 0.5 [rad]] -> [100 [Hz], 400 [Hz]]
    def _conv_theta2freq(self, theta):
        
        return theta * (300.0 / 0.5) + 100.0

    # [0 [rad/s], 3 [rad/s]] -> [100 [Hz], 400 [Hz]]
    def _conv_theta_dot2freq(self, theta_dot):

        return theta_dot * (300.0 / 3.0) + 100.0

    # [0 [Nm], 20 [Nm]] -> [-65 [mV], -55 [mV]]
    def _conv_tau2voltage(self, tau):

        return tau * (10.0 / 20.0) - 65.0

    # returns current[pA]
    def _conv_freq2current(self, freq): # tau_m が十分大きいニューロンを想定

        C_m = 250.0             # [pF]
        V_th = -55.0            # [mV]
        E_L = -70.0             # [mV]
        t_ref = 2.0             # [ms]
        return C_m * (V_th - E_L) * freq / (1000.0 - t_ref * freq)

    def _conv_freq2voltage(self, freq):

        weight = self.output_layer_weight
        tau_syn = 2.0
        C_m = 250.0
        tau_m = 10.0
        E_L = -70.0

        charge_per_spike = weight * tau_syn * math.e / 1000.0 # [pC]
        return freq * charge_per_spike * tau_m / C_m + E_L

    def _conv_voltage2freq(self, voltage):

        weight = self.output_layer_weight
        tau_syn = 2.0
        C_m = 250.0
        tau_m = 10.0
        E_L = -70.0
        
        I = (C_m / tau_m) * (voltage - E_L) # [pA] 
        charge_per_spike = weight * tau_syn * math.e / 1000.0 # [pC]

        return I / charge_per_spike

        
    def _conv_tau2spike_train(self, tau, time_span):

        voltage = self._conv_tau2voltage(tau)

        freq = self._conv_voltage2freq(voltage)
        
        return [round(x, 1) for x in np.linspace(self.total_sim_time + 1.0,
                                                 self.total_sim_time + time_span,
                                                 freq)]


    # [-65 [mv], -55 [mv]] -> [0 [Nm], 20 [Nm]]
    def _update_tau(self, tau1_voltage, tau2_voltage):

        self.tau1 = np.clip((tau1_voltage + 65.0) * (20.0 / 10.0), 0.0, None)
        self.tau2 = np.clip((tau2_voltage + 65.0) * (20.0 / 10.0), 0.0, None)

        
def load(file_path):

    ret = LsmController(1,1,1,1,1)
    ret.load(file_path)
    return ret
    

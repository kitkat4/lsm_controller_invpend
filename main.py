#!/usr/bin/env python
# coding: utf-8



import inverted_pendulum
import lsm_controller

import nest
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import random
import math
import sys
import time
import os



############################################################################

        

def main():
    time_main_start = time.time()
    
    if len(sys.argv) == 3:
        output_dir = sys.argv[1]
        experiment_name = sys.argv[2]
    elif len(sys.argv) == 2:
        output_dir = sys.argv[1]
        experiment_name = "experiment"
    else:
        print "error: specify output directory as a command line argument."
        sys.exit()

    controller = lsm_controller.LsmController(input_neurons_theta_size = 100,
                                              input_neurons_theta_dot_size = 100,
                                              liquid_neurons_size = 10,
                                              readout_neurons_tau1_size = 100,
                                              readout_neurons_tau2_size = 100,
                                              output_layer_weight = 250.0,
                                              thread_num = multiprocessing.cpu_count())


    print_neuron_and_connection_params(controller)

    max_torque = 20.0
    min_torque = -20.0
    Kp = 40.0
    Kd = 9.0
    N_x = 5
    N_y = 5

    
    min_theta = -max_torque/(2*Kp)
    max_theta = -min_torque/(2*Kp)
    min_theta_dot = -max_torque/(2*Kd)
    max_theta_dot = -min_torque/(2*Kd)

    print "cpu count: ", multiprocessing.cpu_count()
    print "Kp and Kd:", Kp, Kd
    print "min and max theta:     ", min_theta, max_theta
    print "min and max theta_dot: ", min_theta_dot, max_theta_dot

    save_figs(controller, "after_0th_training", output_dir, experiment_name, readout_and_output_only = False)

    test_data = [(x, y) for x in np.linspace(min_theta + (max_theta - min_theta)/(N_x + 1), max_theta - (max_theta - min_theta)/(N_x + 1), N_x) for y in np.linspace(min_theta_dot + (max_theta_dot - min_theta_dot)/(N_y + 1), max_theta_dot - (max_theta_dot - min_theta_dot)/(N_y + 1), N_y)]
        
    controller.save(output_dir + "/" + experiment_name + "_after_0th_training.yaml")

    time_calc_rms_error_pd_control_start = time.time()
    rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd, True)
    time_calc_rms_error_pd_control_stop = time.time()
    print "RMS error after 0th training: ", rms_error

        
    # training
    time_training_start = time.time()
    time_net_training = 0.0
    count2 = 1
    for i in range(50000):

        
        theta_train = random.random() * (max_theta - min_theta) + min_theta
        theta_dot_train = random.random() * (max_theta_dot - min_theta_dot) + min_theta_dot
        tau_ref = -Kp * theta_train - Kd * theta_dot_train
        tmp_time = time.time()

        lr = 0.003
        controller.train(theta = theta_train,
                         theta_dot = theta_dot_train,
                         tau1_ref = tau_ref if tau_ref >= 0 else 0.0,
                         tau2_ref = -tau_ref if tau_ref < 0 else 0.0,
                         learning_ratio = lr,
                         momentum_learning_ratio = lr * 0.0,
                         tau1_tolerance = 0.3,
                         tau2_tolerance = 0.3,
                         sim_time = 200.0,
                         filter_size = 100.0)
        time_net_training += time.time() - tmp_time

        if count2 == 20:
            save_figs(controller, "after_" + str(count2) + "th_training", output_dir, experiment_name)

        
        # if count2 % 20 == 0 and count2 <= 200:

        #     controller.save(output_dir + "/" + experiment_name + "_after_" + str(count2) + "th_training.yaml")
        
        if count2 % 100 == 0:
            rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd, True)
            sys.stdout.write("RMS error after " + str(count2) + "th training: " + str(rms_error) + "\n")
            sys.stdout.flush()
            sys.stdout.write("training took " + str(time_net_training) + " [s] (net)\n")
            controller.save(output_dir + "/" + experiment_name + "_after_" + str(count2) + "th_training.yaml")
            save_figs(controller, "after_" + str(count2) + "th_training", output_dir, experiment_name)
            sys.stdout.flush()
            

        count2 += 1
        
    time_training_stop = time.time()
            
    controller.save(output_dir + "/" + experiment_name + "_after.yaml")
    
    rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd)
    print "RMS error after training: ", rms_error


    
    time_main_stop = time.time()
    # sys.stdout.write("calling calc_rms_error_pd_control once took " + str(time_calc_rms_error_pd_control_stop - time_calc_rms_error_pd_control_start) + " [s]\n")
    sys.stdout.write("training took " + str(time_training_stop - time_training_start) + " [s]\n")
    sys.stdout.write("training took " + str(time_net_training) + " [s] (net)\n")
    sys.stdout.write("main took " + str(time_main_stop - time_main_start) + " [s]\n")


    
############################################################################


def calc_rms_error(desired_output, actual_output):

    if len(desired_output) != len(actual_output):
        print "error in calc_rms_error: the length of inputs not match!"
        return None

    size = len(desired_output)
    tmp_sum = 0
    for i in range(size):
        tmp_sum += (desired_output[i] - actual_output[i])**2
    return math.sqrt(tmp_sum / float(size))


def output_with_constant_inputs(controller, theta, theta_dot):

    controller.simulate(500.0, theta, theta_dot, 400.0)
    
    return controller.get_tau()


def calc_rms_error_pd_control(controller, input_list, Kp, Kd, print_message = False):

    actual_output = []
    desired_output = []

    counter = 1
    input_len = len(input_list)
    for itr in input_list:
        if print_message:
            sys.stdout.write("calculating RMS error ... " + str(counter) + "/" + str(input_len) + "    \r")
            sys.stdout.flush()
        actual_output.append(output_with_constant_inputs(controller, itr[0], itr[1]))
        desired_output.append(-Kp * itr[0] - Kd * itr[1])
        counter += 1

    if print_message:
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    return calc_rms_error(desired_output, actual_output)


def save_figs(controller, string, output_dir, experiment_name, suffix = ".eps", readout_and_output_only = True):


    fn_head = output_dir + "/" + experiment_name
    fn_foot = string + suffix

    sim_time = 100.0
    sim_n = 12

    nest.ResetNetwork()
    controller.simulate(sim_time, 0.0, 0.0)
    controller.simulate(sim_time, 0.0, 0.0)
    controller.simulate(sim_time, 0.25, 0.0)
    controller.simulate(sim_time, 0.5, 0.0)
    controller.simulate(sim_time, 0.0, 1.0)
    controller.simulate(sim_time, 0.0, 2.0)
    controller.simulate(sim_time, -0.5, 2.0)
    controller.simulate(sim_time, -0.25, 0.0)
    controller.simulate(sim_time, -0.5, 0.0)
    controller.simulate(sim_time, 0.0, -1.0)
    controller.simulate(sim_time, 0.0, -2.0)
    controller.simulate(sim_time, 0.0, 0.0)
    xticks_time = [sim_time * i for i in range(sim_n + 1)]
    yticks_V_m = [-75.0 + 5.0 * i for i in range(6)]

    controller.lsm.output_layer_tau1.plot_V_m(0, xticks = xticks_time, yticks = yticks_V_m,
                                              time_offset = -controller.total_sim_time + sim_time * sim_n,
                                              file_name = fn_head  + "_out_tau1_V_m_" + fn_foot)
    controller.lsm.output_layer_tau2.plot_V_m(0, xticks = xticks_time, yticks = yticks_V_m,
                                              time_offset = -controller.total_sim_time + sim_time * sim_n,
                                              file_name = fn_head  + "_out_tau2_V_m_" + fn_foot)

    controller.lsm.readout_layer_tau1.raster_plot(xticks = xticks_time,
                                                  yticks = [1, len(controller.lsm.readout_layer_tau1.neurons)] if len(controller.lsm.readout_layer_tau1.neurons) >= 2 else [],
                                                  ylabel = "neuron ID" if len(controller.lsm.readout_layer_tau1.neurons) >= 2 else "",
                                                  xticks_hist = xticks_time,
                                                  gid_offset = -controller.lsm.readout_layer_tau1.neurons[0],
                                                  time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                  hist_binwidth = 50.0,
                                                  file_name = fn_head + "_read_tau1_" + fn_foot)

    controller.lsm.readout_layer_tau2.raster_plot(xticks = xticks_time,
                                                  yticks = [1, len(controller.lsm.readout_layer_tau2.neurons)] if len(controller.lsm.readout_layer_tau2.neurons) >= 2 else [],
                                                  ylabel = "neuron ID" if len(controller.lsm.readout_layer_tau2.neurons) >= 2 else "",
                                                  xticks_hist = xticks_time,
                                                  gid_offset = -controller.lsm.readout_layer_tau2.neurons[0],
                                                  time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                  hist_binwidth = 50.0,
                                                  file_name = fn_head + "_read_tau2_" + fn_foot)

    if not readout_and_output_only:
        controller.lsm.liquid_neurons.raster_plot(xticks = xticks_time,
                                                  yticks = [1, len(controller.lsm.liquid_neurons.neurons)] if len(controller.lsm.liquid_neurons.neurons) >= 2 else [],
                                                  ylabel = "neuron ID" if len(controller.lsm.liquid_neurons.neurons) >= 2 else "",
                                                  xticks_hist = xticks_time,
                                                  gid_offset = -controller.lsm.liquid_neurons.neurons[0],
                                                  time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                  hist_binwidth = 50.0,
                                                  file_name = fn_head + "_liquid_" + fn_foot)
        controller.lsm.input_layer_theta1.raster_plot(xticks = xticks_time,
                                                      yticks = [1, len(controller.lsm.input_layer_theta1.neurons)] if len(controller.lsm.input_layer_theta1.neurons) >= 2 else [],
                                                      ylabel = "neuron ID" if len(controller.lsm.input_layer_theta1.neurons) >= 2 else "",
                                                      xticks_hist = xticks_time,
                                                      gid_offset = -controller.lsm.input_layer_theta1.neurons[0],
                                                      time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                      hist_binwidth = 50.0,
                                                      file_name = fn_head+ "_in_theta1_" + fn_foot)
        controller.lsm.input_layer_theta2.raster_plot(xticks = xticks_time,
                                                      yticks = [1, len(controller.lsm.input_layer_theta2.neurons)] if len(controller.lsm.input_layer_theta2.neurons) >= 2 else [],
                                                      ylabel = "neuron ID" if len(controller.lsm.input_layer_theta2.neurons) >= 2 else "",
                                                      xticks_hist = xticks_time,
                                                      gid_offset = -controller.lsm.input_layer_theta2.neurons[0],
                                                      time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                      hist_binwidth = 50.0,
                                                      file_name = fn_head + "_in_theta2_" + fn_foot)
        controller.lsm.input_layer_theta_dot1.raster_plot(xticks = xticks_time,
                                                          yticks = [1, len(controller.lsm.input_layer_theta_dot1.neurons)] if len(controller.lsm.input_layer_theta_dot1.neurons) >= 2 else [],
                                                          ylabel = "neuron ID" if len(controller.lsm.input_layer_theta_dot1.neurons) >= 2 else "",
                                                          xticks_hist = xticks_time,
                                                          gid_offset = -controller.lsm.input_layer_theta_dot1.neurons[0],
                                                          time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                          hist_binwidth = 50.0,
                                                          file_name = fn_head + "_in_theta_dot1_" + fn_foot)
        controller.lsm.input_layer_theta_dot2.raster_plot(xticks = xticks_time,
                                                          yticks = [1, len(controller.lsm.input_layer_theta_dot2.neurons)] if len(controller.lsm.input_layer_theta_dot2.neurons) >= 2 else [],
                                                          ylabel = "neuron ID" if len(controller.lsm.input_layer_theta_dot2.neurons) >= 2 else "",
                                                          xticks_hist = xticks_time,
                                                          gid_offset = -controller.lsm.input_layer_theta_dot2.neurons[0],
                                                          time_offset = -controller.total_sim_time + sim_time * sim_n,
                                                          hist_binwidth = 50.0,
                                                          file_name = fn_head + "_in_theta_dot2_" + fn_foot)




def print_neuron_and_connection_params(controller):

    conns_input_theta1_to_liquid = len(nest.GetConnections(source = controller.lsm.input_layer_theta1.neurons, target = controller.lsm.liquid_neurons.neurons))
    conns_input_theta2_to_liquid = len(nest.GetConnections(source = controller.lsm.input_layer_theta2.neurons, target = controller.lsm.liquid_neurons.neurons))
    conns_input_theta_dot1_to_liquid = len(nest.GetConnections(source = controller.lsm.input_layer_theta_dot1.neurons, target = controller.lsm.liquid_neurons.neurons))
    conns_input_theta_dot2_to_liquid = len(nest.GetConnections(source = controller.lsm.input_layer_theta_dot2.neurons, target = controller.lsm.liquid_neurons.neurons))
    conns_inside_liquid = len(nest.GetConnections(source = controller.lsm.liquid_neurons.neurons, target = controller.lsm.liquid_neurons.neurons))
    conns_liquid_to_read_tau1 = len(nest.GetConnections(source = controller.lsm.liquid_neurons.neurons, target = controller.lsm.readout_layer_tau1.neurons))
    conns_liquid_to_read_tau2 = len(nest.GetConnections(source = controller.lsm.liquid_neurons.neurons, target = controller.lsm.readout_layer_tau2.neurons))
    sys.stdout.write("\nnumber of neurons:")
    sys.stdout.write("\n    input theta1    : " + str(len(controller.lsm.input_layer_theta1.neurons)))
    sys.stdout.write("\n    input theta2    : " + str(len(controller.lsm.input_layer_theta2.neurons)))
    sys.stdout.write("\n    input theta_dot1: " + str(len(controller.lsm.input_layer_theta_dot1.neurons)))
    sys.stdout.write("\n    input theta_dot2: " + str(len(controller.lsm.input_layer_theta_dot2.neurons)))
    sys.stdout.write("\n    liquid          : " + str(len(controller.lsm.liquid_neurons.neurons)))
    sys.stdout.write("\n    readout tau1    : " + str(len(controller.lsm.readout_layer_tau1.neurons)))
    sys.stdout.write("\n    readout tau2    : " + str(len(controller.lsm.readout_layer_tau2.neurons)))
    sys.stdout.write("\n\nnumber of connections:")
    sys.stdout.write("\n    input theta1 to liquid    : " + str(conns_input_theta1_to_liquid))
    sys.stdout.write("\n    input theta2 to liquid    : " + str(conns_input_theta2_to_liquid))
    sys.stdout.write("\n    input theta_dot1 to liquid: " + str(conns_input_theta_dot1_to_liquid))
    sys.stdout.write("\n    input theta_dot2 to liquid: " + str(conns_input_theta_dot2_to_liquid))
    sys.stdout.write("\n    inside liquid             : " + str(conns_inside_liquid))
    sys.stdout.write("\n    liquid to readout tau1    : " + str(conns_liquid_to_read_tau1))
    sys.stdout.write("\n    liquid to readout tau2    : " + str(conns_liquid_to_read_tau2))
    sys.stdout.write("\n    total                     : " + str(conns_input_theta1_to_liquid + conns_input_theta2_to_liquid + conns_input_theta_dot1_to_liquid + conns_input_theta_dot2_to_liquid + conns_inside_liquid + conns_liquid_to_read_tau1 + conns_liquid_to_read_tau2))
    sys.stdout.write("\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()



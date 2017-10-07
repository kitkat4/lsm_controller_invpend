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

    data = np.zeros(2000)

    for i in range(2000):
        controller.simulate(1.0, theta, theta_dot)
        data[i] = controller.get_tau()

    return data[500:1999].mean()


def calc_rms_error_pd_control(controller, input_list, Kp, Kd):

    actual_output = []
    desired_output = []

    for itr in input_list:
        actual_output.append(output_with_constant_inputs(controller, itr[0], itr[1]))
        desired_output.append(-Kp * itr[0] - Kd * itr[1])

    return calc_rms_error(desired_output, actual_output)


if __name__ == "__main__":

    controller = lsm_controller.LsmController(input_neurons_theta_size = 10,
                                              input_neurons_theta_dot_size = 10,
                                              liquid_neurons_size = 300,
                                              readout_neurons_tau1_size = 5,
                                              readout_neurons_tau2_size = 5,
                                              output_layer_weight = 100.0,
                                              thread_num = multiprocessing.cpu_count())

    print "cpu count: ", multiprocessing.cpu_count()
    
    pend = inverted_pendulum.InvertedPendulum(mass = 1.0,
                                              length = 1.0,
                                              theta_0 = 1.0,
                                              theta_dot_0 = 0.0)

    # result1_prev = np.zeros(2000)
    # result2_prev = np.zeros(2000)
    # result1 = np.zeros(2000)
    # result2 = np.zeros(2000)
    

    # for time in range(2000):

    #     controller.simulate(1.0, 1.0, 0.0)
    #     result1_prev[time] = controller.tau1
    #     result2_prev[time] = controller.tau2

    # # before learning
    # for time in range(1000):

    #     theta = pend.theta
    #     theta_dot = pend.theta_dot
    #     controller.simulate(1.0, theta, theta_dot)
    #     pend.simulate_one_step(controller.get_tau(), 0.001)
    #     print time
        
    # pend.plot()

    training_data = [(-2.0, 0.0),
                     (-1.5, -3.0),
                     (-1.5, 0.0),
                     (-1.5, 3.0),
                     (-1.0, -6.0),
                     (-1.0, -3.0),
                     (-1.0, 0.0),
                     (-1.0, 3.0),
                     (-1.0, 6.0),
                     (-0.5, -6.0),
                     (-0.5, -3.0),
                     (-0.5, 0.0),
                     (-0.5, 3.0),
                     (-0.5, 6.0),
                     (0.0, -9.0),
                     (0.0, -6.0),
                     (0.0, -3.0),
                     (0.0, 0.0),
                     (0.0, 3.0),
                     (0.0, 6.0),
                     (0.0, 9.0),
                     (2.0, 0.0),
                     (1.5, -3.0),
                     (1.5, 0.0),
                     (1.5, 3.0),
                     (1.0, -6.0),
                     (1.0, -3.0),
                     (1.0, 0.0),
                     (1.0, 3.0),
                     (1.0, 6.0),
                     (0.5, -6.0),
                     (0.5, -3.0),
                     (0.5, 0.0),
                     (0.5, 3.0),
                     (0.5, 6.0)]
    
    calc_rms_error_pd_control(controller, training_data, 40.0, 9.0)
    
    random.shuffle(training_data)

    print training_data

    exit()
    
    for itr in training_data:
    
        controller.train(theta = 1.0,
                         theta_dot = 0.0,
                         tau1_ref = 0.0,
                         tau2_ref = 40.0,
                         update_num = 1,  
                         sim_time = 1000.0,
                         print_message = False)

    

    for time in range(2000):

        # theta = pend.theta
        # theta_dot = pend.theta_dot
        controller.simulate(1.0, 1.0, 0.0)
        result1[time] = controller.tau1
        result2[time] = controller.tau2
        # pend.simulate_one_step(controller.get_tau(), 0.001)

    print "mean output before training: ", result1_prev.mean() - result2_prev.mean()
    print "mean output after  training: ", result1.mean() - result2.mean()


    plt.figure()
    plt.plot(result1_prev, 'b.')
    plt.plot(result1, 'r.')
    plt.title("tau1")
    plt.figure()
    plt.plot(result2_prev, 'b.')
    plt.plot(result2, 'r.')
    plt.title("tau2")

    plt.show()
    
    # pend.plot()

    # for debugging
    liq = controller.lsm.liquid_neurons
    in_theta = controller.lsm.input_layer_theta
    in_theta_dot = controller.lsm.input_layer_theta_dot
    readout1 = controller.lsm.readout_layer_tau1
    readout2 = controller.lsm.readout_layer_tau2
    output1 = controller.lsm.output_layer_tau1
    output2 = controller.lsm.output_layer_tau2
    


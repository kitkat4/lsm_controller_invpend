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

    controller.simulate(600.0, theta, theta_dot, 450.0)
    
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


if __name__ == "__main__":


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

    controller = lsm_controller.LsmController(input_neurons_theta_size = 5,
                                              input_neurons_theta_dot_size = 5,
                                              liquid_neurons_size = 300,
                                              readout_neurons_tau1_size = 1,
                                              readout_neurons_tau2_size = 1,
                                              output_layer_weight = 100.0,
                                              thread_num = multiprocessing.cpu_count())

    print "cpu count: ", multiprocessing.cpu_count()

    controller.simulate(1000.0, 0.0, 0.0)
    controller.simulate(1000.0, 1.0, 0.0)
    controller.simulate(1000.0, 1.0, 5.0)
    controller.simulate(1000.0, -1.0, 0.0)
    controller.simulate(1000.0, -1.0, -5.0)
    controller.lsm.output_layer_tau1.plot(0)
    controller.lsm.output_layer_tau2.plot(0)
    
    pend = inverted_pendulum.InvertedPendulum(mass = 1.0,
                                              length = 1.0,
                                              theta_0 = 1.0,
                                              theta_dot_0 = 0.0)

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
    
    test_data = [(x, y) for x in np.linspace(min_theta, max_theta, N_x) for y in np.linspace(min_theta_dot, max_theta_dot, N_y)]
        
    controller.save(output_dir + "/" + experiment_name + "_before.yaml")    

    time_calc_rms_error_pd_control_start = time.time()
    rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd)
    time_calc_rms_error_pd_control_stop = time.time()
    print "RMS error before training: ", rms_error

        
    # training
    time_training_start = time.time()
    count2 = 1
    for i in range(3000):

        theta_train = random.random() * (max_theta - min_theta) + min_theta
        theta_dot_train = random.random() * (max_theta_dot - min_theta_dot) + min_theta_dot
        tau_ref = -Kp * theta_train - Kd * theta_dot_train
        controller.train(theta = theta_train,
                         theta_dot = theta_dot_train,
                         tau1_ref = tau_ref if tau_ref >= 0 else 0.0,
                         tau2_ref = -tau_ref if tau_ref < 0 else 0.0,
                         update_num = 1,  
                         sim_time = 200.0,
                         print_message = False)

        # sys.stdout.write("train (" + str(theta_train) + ", " + str(theta_dot_train) + ")\n")
        
        if count2 % 10 == 0:
            rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd)
            print "RMS error after " + str(count2) + "th training: ", rms_error

        count2 += 1

    time_training_stop = time.time()
            
    controller.save(output_dir + "/" + experiment_name + "_after.yaml")
    
    rms_error = calc_rms_error_pd_control(controller, test_data, Kp, Kd)
    print "RMS error after training: ", rms_error


    # result1_prev = np.zeros(2000)
    # result2_prev = np.zeros(2000)
    # result1 = np.zeros(2000)
    # result2 = np.zeros(2000)
    

    # for i in range(2000):

    #     controller.simulate(1.0, 1.0, 0.0)
    #     result1_prev[i] = controller.tau1
    #     result2_prev[i] = controller.tau2
    

    # time_training_start = time.time()
    # controller.train(1.0, 0.0, 40.0, 0.0, 30, 1000.0)
    # time_training_stop = time.time()

    
    # for i in range(2000):

    #     # theta = pend.theta
    #     # theta_dot = pend.theta_dot
    #     controller.simulate(1.0, 1.0, 0.0)
    #     result1[i] = controller.tau1
    #     result2[i] = controller.tau2
    #     # pend.simulate_one_step(controller.get_tau(), 0.001)

    # print "mean output before training: ", result1_prev.mean() - result2_prev.mean()
    # print "mean output after  training: ", result1.mean() - result2.mean()


    # plt.figure()
    # plt.plot(result1_prev, 'b.')
    # plt.plot(result1, 'r.')
    # plt.title("tau1")
    # plt.figure()
    # plt.plot(result2_prev, 'b.')
    # plt.plot(result2, 'r.')
    # plt.title("tau2")

    # plt.show()
    

##########################################
    
    time_main_stop = time.time()
    # sys.stdout.write("calling calc_rms_error_pd_control once took " + str(time_calc_rms_error_pd_control_stop - time_calc_rms_error_pd_control_start) + " [s]\n")
    sys.stdout.write("training took " + str(time_training_stop - time_training_start) + " [s]\n")
    sys.stdout.write("main took " + str(time_main_stop - time_main_start) + " [s]\n")

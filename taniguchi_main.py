#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum
import lsm_controller

import nest
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    controller = lsm_controller.LsmController(input_neurons_theta_size = 10,
                                              input_neurons_theta_dot_size = 10,
                                              liquid_neurons_size = 300,
                                              readout_neurons_tau1_size = 5,
                                              readout_neurons_tau2_size = 5,
                                              output_layer_weight = 100.0,
                                              thread_num = 8)

    pend = inverted_pendulum.InvertedPendulum(mass = 1.0,
                                              length = 1.0,
                                              theta_0 = 1.0,
                                              theta_dot_0 = 0.0)

    result1_prev = np.zeros(2000)
    result2_prev = np.zeros(2000)
    result1 = np.zeros(2000)
    result2 = np.zeros(2000)

    Kp = 40.0
    Kd = 9.0
    theta_goal = 0.0
    theta_dot_goal = 0.0

    for time in range(5000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        torque = Kp*(theta_goal-theta)+Kd*(theta_dot_goal-theta_dot)
        pend.simulate_one_step(torque , 0.001)


    for time in range(2000):
        theta = pend.theta
        theta_dot = pend.theta_dot

        controller.simulate(1.0, theta, theta_dot)

        result1_prev[time] = controller.tau1
        result2_prev[time] = controller.tau2

    # # before learning
    # for time in range(1000):

    #     theta = pend.theta
    #     theta_dot = pend.theta_dot
    #     controller.simulate(1.0, theta, theta_dot)
    #     pend.simulate_one_step(controller.get_tau(), 0.001)
    #     print time

    # pend.plot()

    tau_max = max(pend.torque_data) - min(pend.torque_data)

    epoch = 1

    for i in range(epoch):

        for time in range(2000):
            tau1_ref = (tau_max + pend.torque_data[time]) / 2
            tau2_ref = (tau_max - pend.torque_data[time]) / 2

            controller.train(theta = pend.theta_data[time],
                             theta_dot = pend.theta_dot_data[time],
                             tau1_ref = tau1_ref,
                             tau2_ref = tau2_ref,
                             update_num = 1,
                             sim_time = 1.0,
                             print_message = True)

            sys.stdout.write("training: " + str(time+1) + "/" + str(2000) +  "    \r")
            sys.stdout.flush()



    for time in range(2000):
        theta = pend.theta
        theta_dot = pend.theta_dot

        controller.simulate(1.0, theta, theta_dot)

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

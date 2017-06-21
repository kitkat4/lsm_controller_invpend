#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum
import lsm_controller

import nest
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    controller = lsm_controller.LsmController(input_neurons_theta_size = 30,
                                              input_neurons_theta_dot_size = 30,
                                              liquid_neurons_size = 300,
                                              readout_neurons_tau1_size = 5,
                                              readout_neurons_tau2_size = 5,
                                              thread_num = 12)
    
    pend = inverted_pendulum.InvertedPendulum(mass = 1.0,
                                              length = 1.0,
                                              theta_0 = 1.0,
                                              theta_dot_0 = 0.0)

    
    # # before learning
    # for time in range(1000):

    #     theta = pend.theta
    #     theta_dot = pend.theta_dot
    #     controller.simulate(1.0, theta, theta_dot)
    #     pend.simulate_one_step(controller.get_tau(), 0.001)
    #     print time
        
    # pend.plot()

    # controller.train(theta = 1.0,
    #                  theta_dot = 0.0,
    #                  tau1_ref = 0.0,
    #                  tau2_ref = 40.0,
    #                  update_num = 10,
    #                  sim_time = 1000.0,
    #                  print_message = True)

    result = np.zeros(5000)

    for time in range(5000):

        # theta = pend.theta
        # theta_dot = pend.theta_dot
        controller.simulate(1.0, 1.0, 0.0)
        result[time] = controller.get_tau()
        # pend.simulate_one_step(controller.get_tau(), 0.001)

    print result.mean()

    plt.plot(result, '.')
    plt.show()
    
    # pend.plot()

    


    
    


#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd
import lsm_controller

import threading
import sys
import time

if __name__ == "__main__":

    yaml_path = sys.argv[1]
    filter_size = float(sys.argv[2])
    
    # pend = ivtpnd.InvertedPendulum(mass = 60.0,
    #                                length = 1.7,
    #                                theta_0 = 0.05)
    # Kp = 600.0
    # Kd = 150.0

    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   theta_0 = 0.3)
    Kp = 40.0
    Kd = 9.0

    controller = lsm_controller.load(yaml_path)

    theta_goal = 0.0
    theta_dot_goal = 0.0

    for t in range(5000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        controller.simulate(1.0, theta, theta_dot, filter_size) # こっちは単位が[ms]
        torque = controller.get_tau()
        # if t == 2500:
        #     controller.lsm.output_layer_tau1.plot_V_m(0)
        # torque = Kp*(theta_goal-theta)+Kd*(theta_dot_goal-theta_dot)
        
        pend.simulate_one_step(torque , 0.001)


    pend.plot()

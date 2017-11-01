#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd
import lsm_controller

import threading
import sys
import time
import numpy as np

if __name__ == "__main__":

    if len(sys.argv) >= 3:
        controller = lsm_controller.load(sys.argv[1])
        filter_size = float(sys.argv[2])
    elif len(sys.argv) == 2:
        controller = lsm_controller.load(sys.argv[1])
        filter_size = 1.0
    else:                       # use PD controller
        controller = None
        filter_size = None
        
    
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

    theta_goal = 0.0
    theta_dot_goal = 0.0

    for t in range(15000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        if controller is not None:
            controller.simulate(1.0, theta, theta_dot, filter_size) # こっちは単位が[ms]
            torque = controller.get_tau()
        else:
            torque = -Kp * (theta - theta_goal) - Kd * (theta_dot-theta_dot_goal)
            torque += np.random.randn()
        
        pend.simulate_one_step(torque , 0.001)


    pend.plot()

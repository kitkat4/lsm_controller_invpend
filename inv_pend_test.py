#!/usr/bin/env python 
# coding: utf-8

import inverted_pendulum as ivtpnd

import threading
import time

if __name__ == "__main__":
    
    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   theta_0 = 2.0)

    
    # pend.simulate(torque = 0.0,
    #               timestep = 0.001,
    #               num_steps = 5000)

    for time in range(5000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        pend.simulate_one_step(-40*theta-9*theta_dot , 0.001)


    pend.plot()



#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd

import threading
import time

import math

if __name__ == "__main__":

    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   theta_0 = 1.0)


    # pend.simulate(torque = 0.0,
    #               timestep = 0.001,
    #               num_steps = 5000)

    theta_goal = math.asin(1)
    theta_dot_goal = 0.0
    Kp = 75.0
    Kd = 3.0

    for time in range(5000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        pend.simulate_one_step(Kp * (theta_goal - theta) + Kd * (theta_dot_goal - theta_dot) , 0.001)


    pend.plot()

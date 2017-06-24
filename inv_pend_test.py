#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd

import threading
import time

import math

if __name__ == "__main__":

    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   phi_0 = 0.2)


    # pend.simulate(torque = 0.0,
    #               timestep = 0.001,
    #               num_steps = 5000)

    phi_goal = 0.0
    phi_dot_goal = 0.0
    phi_integral = 0.0
    Kp = 58.8  # Kp - mgl >= 0 より Kp >= 9.8
    Kd = 3.0
    Ki = 0.0
    timestep = 0.001

    for time in range(5000):
        phi = pend.phi
        phi_dot = pend.phi_dot
        phi_integral += phi*0.001
        #pend.simulate_one_step(Kp * (phi_goal - phi) + Kd * (phi_dot_goal - phi_dot + Ki * (phi_goal*time*0.001 - phi_integral)) , 0.001)
        pend.simulate_one_step(Kp, Kd, time, timestep)

    pend.plot()

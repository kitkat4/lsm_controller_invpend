#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd

import threading
import time

if __name__ == "__main__":

    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   theta_0 = 0.5)


    # pend.simulate(torque = 0.0,
    #               timestep = 0.001,
    #               num_steps = 5000)

    Kp = 40.0
    Kd = 9.0
    theta_goal = 0.0
    theta_dot_goal = 0.0

    for time in range(5000):
        theta = pend.theta
        theta_dot = pend.theta_dot
        torque = Kp*(theta_goal-theta)+Kd*(theta_dot_goal-theta_dot)
        pend.simulate_one_step(torque , 0.001)


    pend.plot()

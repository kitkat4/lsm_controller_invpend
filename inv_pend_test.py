#!/usr/bin/env python
# coding: utf-8

import inverted_pendulum as ivtpnd

import threading
import time

if __name__ == "__main__":

    # pend = ivtpnd.InvertedPendulum(mass = 60.0,
    #                                length = 1.7,
    #                                theta_0 = 0.05)
    # Kp = 600.0
    # Kd = 150.0

    pend = ivtpnd.InvertedPendulum(mass = 1.0,
                                   length = 1.0,
                                   theta_0 = 1.0)
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

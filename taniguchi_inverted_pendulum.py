# coding : utf-8


# import numpy as np
import matplotlib.pyplot as plt
import math


# theta_0[rad] equals zero when the pendulum is standing vertically. CW positive
class InvertedPendulum:

    def __init__(self,
                 mass,
                 length,
                 theta_0,
                 theta_dot_0 = 0.0,
                 gravitational_acc = 9.8,
                 store_old_data = True):

        self.m = mass
        self.l = length
        self.theta = theta_0
        self.theta_dot = theta_dot_0
        self.g = gravitational_acc

        self.store_old_data = store_old_data
        self.theta_data = [self.theta] if store_old_data else []
        self.theta_dot_data = [self.theta_dot] if store_old_data else []
        self.torque_data = []
        self.time_data = [0.0]

    def simulate(self, torque, timestep, num_steps):

        for i in range(num_steps):
            self.simulate_one_step(torque, timestep)


    # Simulate using RK4 (Runge-Kutta).
    # torque[Nm]
    # timestep[s]
    def simulate_one_step(self, Kp, Kd, theta_goal, theta_dot_goal, timestep):

        tmp1 = 3.0 / (self.m * self.l * self.l)
        tmp2 = 3.0 * self.g / (2.0 * self.l)
        torque = self.torque(Kp, Kd, theta_goal, theta_dot_goal, self.theta, self.theta_dot)

        # k*: approximation of theta_dot
        # l*: approximation of theta_ddot
        k1 = self.theta_dot
        l1 = tmp1*self.torque(Kp, Kd, theta_goal, theta_dot_goal, self.theta, self.theta_dot) + tmp2*math.sin(self.theta)

        k2 = self.theta_dot + timestep*l1/2.0
        l2 = tmp1*self.torque(Kp, Kd, theta_goal, theta_dot_goal, self.theta+k1*timestep/2.0, self.theta_dot+l1*timestep/2.0) + tmp2*math.sin(self.theta+timestep*k1/2.0)

        k3 = self.theta_dot + timestep*l2/2.0
        l3 = tmp1*self.torque(Kp, Kd, theta_goal, theta_dot_goal, self.theta+k2*timestep/2.0, self.theta_dot+l2*timestep/2.0) + tmp2*math.sin(self.theta+timestep*k2/2.0)

        k4 = self.theta_dot + timestep*l3
        l4 = tmp1*self.torque(Kp, Kd, theta_goal, theta_dot_goal, self.theta+k3*timestep, self.theta_dot+l3*timestep) + tmp2*math.sin(self.theta+timestep*k3)

        # update
        self.theta = self.theta + timestep * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        self.theta_dot = self.theta_dot + timestep * (l1 + 2*l2 + 2*l3 + l4) / 6.0

        if self.store_old_data:
            self.theta_data.append(self.theta)
            self.theta_dot_data.append(self.theta_dot)
            self.torque_data.append(torque)
            self.time_data.append(self.time_data[-1] + timestep)

    def torque(self, Kp, Kd, theta_goal, theta_dot_goal, theta, theta_dot):
        return Kp*(theta_goal-theta)+Kd*(theta_dot_goal-theta_dot)


    def plot(self):

        if not self.store_old_data:
            return

        plt.figure()
        plt.plot(self.time_data, self.theta_data,
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("theta [rad]")
        plt.figure()
        plt.plot(self.time_data, self.theta_dot_data,
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("theta dot [rad/s]")
        plt.figure()
        plt.plot(self.time_data, self.torque_data + [self.torque_data[-1]],
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("torque [Nm]")

        plt.show()

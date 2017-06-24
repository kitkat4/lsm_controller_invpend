# coding : utf-8


# import numpy as np
import matplotlib.pyplot as plt
import math


# phi_0[rad] equals zero when the pendulum is standing vertically. CW positive
class InvertedPendulum:

    def __init__(self,
                 mass,
                 length,
                 phi_0,
                 phi_dot_0 = 0.0,
                 gravitational_acc = 9.8,
                 store_old_data = True):

        self.m = mass
        self.l = length
        self.phi = phi_0
        self.phi_dot = phi_dot_0
        self.g = gravitational_acc
        self.J = mass * length * length / 3.0

        self.store_old_data = store_old_data
        self.phi_data = [self.phi] if store_old_data else []
        self.phi_dot_data = [self.phi_dot] if store_old_data else []
        self.torque_data = []
        self.time_data = [0.0]

    def simulate(self, torque, timestep, num_steps):

        for i in range(num_steps):
            self.simulate_one_step(torque, timestep)


    # Simulate using RK4 (Runge-Kutta).
    # torque[Nm]
    # timestep[s]
    def simulate_one_step(self, Kp, Kd, num, timestep):

        time = timestep*num
        torque  = Kp * self.phi + Kd * self.phi_dot
        ganma = Kd / (self.J * 2)
        omega = math.sqrt((Kp - self.m * self.g * self.l) / self.J)
        if ganma > omega :
            ganma2omega2 = math.sqrt(ganma*ganma - omega*omega)
            if time == 0.0:
                A = (self.phi + self.phi_dot * (ganma2omega2 + ganma)) / (2 * ganma2omega2)
                B = self.phi - A
            else :
                A = (self.phi_data[0] + self.phi_dot_data[0] * (ganma2omega2 + ganma)) / (2 * ganma2omega2)
                B = self.phi_data[0] - A

            self.phi = math.exp(-ganma * time) * (A * math.exp(ganma2omega2 * time) + B * (math.exp(-ganma2omega2 * time)))
            self.phi_dot = math.exp(-ganma * time) * (A * (ganma2omega2 - ganma) * math.exp(ganma2omega2 * time) - B * (ganma2omega2 + ganma) * math.exp(-ganma2omega2 * time))

        elif ganma < omega :
            ganma2omega2 = math.sqrt(omega*omega - ganma*ganma)
            if time == 0.0:
                A = self.phi
                B = (self.phi_dot + ganma*self.phi) / ganma2omega2
            else :
                A = self.phi_data[0]
                B = (self.phi_dot_data[0] + ganma*self.phi_data[0]) / ganma2omega2

            self.phi = math.exp(-ganma * time) * (A*math.cos(ganma2omega2*time) + B*math.sin(ganma2omega2*time))
            self.phi_dot = math.exp(-ganma * time) * ((-ganma*A + B*ganma2omega2)*math.cos(ganma2omega2*time) + (-ganma*B - A*ganma2omega2)*math.sin(ganma2omega2*time))

        else :
            ganma2omega2 = 0.0
            if time == 0.0:
                A = self.phi
                B = self.phi_dot + ganma*self.phi
            else :
                A = self.phi_data[0]
                B = self.phi_dot_data[0] + ganma*self.phi_data[0]

            self.phi = math.exp(-ganma * time) * (A + B*time)
            self.phi_dot = math.exp(-ganma * time) *(B - ganma*A - ganma*B*time)

        # k*: approximation of phi_dot
        # l*: approximation of phi_ddot
        """
        k1 = self.phi_dot
        l1 = tmp1 + tmp2 * math.sin(self.phi)

        k2 = self.phi_dot + timestep * l1 / 2.0
        l2 = tmp1 + tmp2 * math.sin(self.phi + timestep * k1 / 2.0)

        k3 = self.phi_dot + timestep * l2 / 2.0
        l3 = tmp1 + tmp2 * math.sin(self.phi + timestep * k2 / 2.0)

        k4 = self.phi_dot + timestep * l3
        l4 = tmp1 + tmp2 * math.sin(self.phi + timestep * k3)

        # update
        self.phi = self.phi + timestep * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        self.phi_dot = self.phi_dot + timestep * (l1 + 2*l2 + 2*l3 + l4) / 6.0
        """

        if self.store_old_data:
            self.phi_data.append(self.phi)
            self.phi_dot_data.append(self.phi_dot)
            self.torque_data.append(torque)
            self.time_data.append(self.time_data[-1] + timestep)


    def plot(self):

        if not self.store_old_data:
            return

        plt.figure()
        plt.plot(self.time_data, self.phi_data,
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("phi [rad]")
        plt.figure()
        plt.plot(self.time_data, self.phi_dot_data,
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("phi dot [rad/s]")
        plt.figure()
        plt.plot(self.time_data, self.torque_data + [self.torque_data[-1]],
                 '.', markersize = 3)
        plt.xlabel("time [s]")
        plt.ylabel("torque [Nm]")

        plt.show()

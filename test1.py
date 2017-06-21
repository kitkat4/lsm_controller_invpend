# coding = utf-8

import nest
import matplotlib.pyplot as plt

import numpy as np

import math

if __name__ == "__main__":

    weight = 500.0
    tau_syn = 2.0
    C_m = 250.0
    tau_m = 10.0
    E_L = -70.0
    desired_voltage = 40.0 * 0.1 - 60.0
    
    time_span = 1000.0
    
    def desired_train(voltage):

        I = (C_m / tau_m) * (voltage - E_L) # [pA] 
        charge_per_spike = weight * tau_syn * math.e / 1000.0 # [pC]
        freq = I / charge_per_spike
        print "desired frequency: ", freq
        return [round(x, 1) for x in np.linspace(1, time_span, freq)]
        
    
    generator = nest.Create("spike_generator",
                            params = {"spike_times": desired_train(desired_voltage)})
    neuron = nest.Create("iaf_psc_alpha", params = {"V_th": float(10**10)})

    meter = nest.Create("multimeter",
                        params = {"withtime": True,
                                  "record_from": ["V_m"]})
    
    detector = nest.Create("spike_detector",
                           params = {"withgid": True,
                                     "withtime": True})

    nest.Connect(generator, neuron,
                 {"rule": "all_to_all"},
                 {"model": "static_synapse", "weight": weight})
    nest.Connect(meter, neuron)
    nest.Connect(neuron, detector)

    nest.Simulate(1000.0)

    result_meter = nest.GetStatus(meter)[0]
    result_detector = nest.GetStatus(detector)[0]

    plt.figure()
    plt.plot(result_meter["events"]["times"],
             result_meter["events"]["V_m"])

    plt.figure()
    plt.plot(result_detector["events"]["times"],
             result_detector["events"]["senders"],
             '.')

    print "mean voltage: ", sum(result_meter["events"]["V_m"]) / len(result_meter["events"]["V_m"])

    plt.show()


    

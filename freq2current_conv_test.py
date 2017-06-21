# coding = utf-8

import nest
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    freq = 499.0
    sim_time = 100000.0

    C_m = 250.0             # [pF]
    V_th = -55.0            # [mV]
    E_L = -70.0             # [mV]
    t_ref = 2.0
    # current = C_m * (V_th - E_L) * freq / 1000.0
    current = C_m * (V_th - E_L) * freq / (1000.0 - t_ref * freq)


    neuron = nest.Create("iaf_psc_alpha",
                         params = {"I_e": current,
                                   "tau_m": float(10**100)}) # float("inf")だと正常動作しなかった
                                   # "V_m": -60.0})

    meter = nest.Create("multimeter",
                             params = {"withtime": True,
                                       "record_from": ["V_m"]})
    
    detector = nest.Create("spike_detector",
                           params = {"withgid": True,
                                     "withtime": True})

    nest.Connect(meter, neuron)
    nest.Connect(neuron, detector)

    nest.Simulate(sim_time)

    result_meter = nest.GetStatus(meter)[0]
    result_detector = nest.GetStatus(detector)[0]

    print "freq_des: ", freq, ", current: ", current
    freq_act = 1000 * len(result_detector["events"]["times"]) / sim_time
    print "freq_act: ", freq_act, ", error: ", ((freq_act - freq) / freq) * 100.0 , "%"

    plt.figure()
    plt.plot(result_meter["events"]["times"],
             result_meter["events"]["V_m"])

    plt.figure()
    plt.plot(result_detector["events"]["times"],
             result_detector["events"]["senders"],
             '.')

    plt.show()

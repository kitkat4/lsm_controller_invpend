from math import *


def resume(input_spike_train, output_spike_train, desire_spike_train):

    a = 0.025
    A_positive = 4.0 * 10**-10
    A_negative =  (-1.0) * 0.1 * A_positive
    tau = 2.0 * 10**-3

    def Learning_Window(s):
        if s >= 0:
            return A_positive * exp((-1.0)*s / tau)
        else:
            return A_negative * exp(s / tau)

    delta_w = 0
    
    ## compare desire with input
    for desire_spike_time in desire_spike_train:
        delta_w += a
        for input_spike_time in input_spike_train:
            s = desire_spike_time - input_spike_time
            if s > 100:
                continue
            elif s < -100:
                break
            else:
                delta_w += Learning_Window(s*10**-3)# convert [ms] to [s] 

    ## compare output with input
    for output_spike_time in output_spike_train:
        delta_w -= a
        for input_spike_time in input_spike_train:
            s = output_spike_time - input_spike_time
            if s > 100:
                continue
            elif s < -100:
                break
            else:
                delta_w -= Learning_Window(s*10**-3)# convert [ms] to [s] 

    return delta_w




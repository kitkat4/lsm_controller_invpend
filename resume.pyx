from math import *


def resume(double[:] input_spike_train, double[:] output_spike_train, double[:] desire_spike_train):

    cdef double a = 0.025
    cdef double A_positive = 4.0 * 10**-10
    cdef double A_negative =  (-1.0) * 0.1 * A_positive
    cdef double tau = 2.0

    cdef double delta_w = 0

    cdef double desire_spike_time, output_spike_time
    
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
                delta_w += A_positive * exp((-1.0)*s / tau) if s >= 0 else A_negative * exp(s / tau)

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
                delta_w -= A_positive * exp((-1.0)*s / tau) if s >= 0 else A_negative * exp(s / tau)

    return delta_w




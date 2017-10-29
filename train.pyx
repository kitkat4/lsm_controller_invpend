from math import *
import nest
import numpy as np

def resume(double[:] input_spike_train, double[:] output_spike_train, double[:] desire_spike_train,
           double a, double A_positive, double A_negative, double tau):

    # cdef double a = 0.025
    # cdef double A_positive = 4.0 * 10**-10
    # cdef double A_negative =  (-1.0) * 0.1 * A_positive
    # cdef double tau = 2.0

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


# def train(double[:] tau_error, double learning_ratio, double momentum_learning_ratio, double tolerance, double filter_size, int[:] self_neurons, presynaptic_neurons, previous_delta_w, connected_liquid):

#     cdef int neuron_ix, pre_ix, spike_num, pre_n, neuron_num
#     cdef double present_weight, tmp_delta_w1, tmp_delta_w2, new_weight
    
#     delta_w = {}

#     neuron_num = len(self_neurons)
    
#     for neuron_ix in range(neuron_num):
#         if tau_error[neuron_ix] > tolerance or tau_error[neuron_ix] < -tolerance:
#             for pre_ix in presynaptic_neurons[neuron_ix]:
#                 spike_num = connected_liquid.num_of_spikes(pre_ix, filter_size)
#                 pre_n = connected_liquid.neurons[pre_ix]
#                 conn = nest.GetConnections(source = [pre_n], target = [self_neurons[neuron_ix]])
#                 present_weight = nest.GetStatus(conn)[0]["weight"]
#                 tmp_delta_w1 = (learning_ratio * abs(tau_error[neuron_ix]) * spike_num * 1000.0 / filter_size) * (-1 if tau_error[neuron_ix] > tolerance else 1)
#                 tmp_delta_w2 = momentum_learning_ratio * (previous_delta_w[(neuron_ix, pre_ix)] if (neuron_ix, pre_ix) in previous_delta_w else 0)
#                 new_weight = present_weight + tmp_delta_w1 + tmp_delta_w2
#                 nest.SetStatus(conn, {"weight": new_weight})
#                 delta_w[(neuron_ix, pre_ix)] = new_weight - present_weight
#                 # sys.stdout.write(str(present_weight) + " -> " + str(new_weight) + " (" + str(new_weight - present_weight) + ")\n")

#     return delta_w

    
def train(double tau_error, double learning_ratio, double momentum_learning_ratio, double tolerance, double filter_size, int target_gid, int[:] presynaptic_neurons, conns, previous_delta_w, connected_liquid):


    
    cdef int pre_ix, i
    
    if len(previous_delta_w) < len(presynaptic_neurons):
        previous_delta_w = np.zeros(len(presynaptic_neurons))

    delta_w = np.zeros(len(presynaptic_neurons))

    if len(conns) == 0:
        return delta_w

    if tau_error > tolerance or tau_error < -tolerance:
        
        present_weights = np.array(nest.GetStatus(conns, keys = "weight"))
        new_weights = np.zeros(len(present_weights), dtype = np.float64)
        spike_nums = np.zeros(len(present_weights), dtype = np.int64)
        for i, pre_ix in enumerate(presynaptic_neurons):
            spike_nums[i] = connected_liquid.num_of_spikes(pre_ix, filter_size)
            
        tmp_delta_w1 = (learning_ratio * abs(tau_error) * spike_nums * 1000.0 / filter_size) * (-1 if tau_error > tolerance else 1)
        tmp_delta_w2 = momentum_learning_ratio * previous_delta_w 
        new_weights = present_weights + tmp_delta_w1 + tmp_delta_w2
        nest.SetStatus(conns, [{"weight": nw} for nw in new_weights])
        delta_w = new_weights - present_weights
        # sys.stdout.write(str(present_weight) + " -> " + str(new_weight) + " (" + str(new_weight - present_weight) + ")\n")

    return delta_w

    

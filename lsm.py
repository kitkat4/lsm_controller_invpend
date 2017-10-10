# coding: utf-8

import neuron_layer
import liquid_neurons
import nest
import numpy as np

import yaml

import sys
import copy

class Lsm:

    def __init__(self,
                 input_neurons_theta_size,
                 input_neurons_theta_dot_size,
                 liquid_neurons_size,
                 readout_neurons_tau1_size,
                 readout_neurons_tau2_size,
                 output_layer_weight):

        # create neurons
        self.input_layer_theta = neuron_layer.NeuronLayer(input_neurons_theta_size,
                                                          tau_m = float(10**100))

        self.input_layer_theta_dot = neuron_layer.NeuronLayer(input_neurons_theta_dot_size,
                                                              tau_m = float(10**100))
        
        self.liquid_neurons = liquid_neurons.LiquidNeurons(liquid_neurons_size,
                                                           0.005, 0.25, "iaf_psc_alpha",
                                                           100.0, 500.0, 0.5, 4.0)
        
        self.readout_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size)
        self.readout_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau2_size)

        self.output_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        self.output_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        
        # connect layers
        self.input_layer_theta.connect2liquid(self.liquid_neurons, 0.3, 0.25, 100.0, 500.0, 0.5, 4.0)
        self.input_layer_theta_dot.connect2liquid(self.liquid_neurons, 0.3, 0.25, 100.0, 500.0, 0.5, 4.0)
        self.liquid_neurons.connect(self.readout_layer_tau1, 0.3, 0.25, 100.0, 500.0, 0.5, 4.0)
        self.liquid_neurons.connect(self.readout_layer_tau2, 0.3, 0.25, 100.0, 500.0, 0.5, 4.0)
        self.readout_layer_tau1.connect2layer_one_to_one(self.output_layer_tau1,
                                                         weight = output_layer_weight)
        self.readout_layer_tau2.connect2layer_one_to_one(self.output_layer_tau2,
                                                         weight = output_layer_weight)



        
    def save(self, file_name):

        data = {}               # data to write into a file

        tmp_dict = {}
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "input_layer_theta",
                                         neurons = self.input_layer_theta.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "input_layer_theta_dot",
                                         neurons = self.input_layer_theta_dot.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "liquid_neurons",
                                         neurons = self.liquid_neurons.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "readout_layer_tau1",
                                         neurons = self.readout_layer_tau1.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "readout_layer_tau2",
                                         neurons = self.readout_layer_tau2.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "output_layer_tau1",
                                         neurons = self.output_layer_tau1.neurons)
        self._append_neuron_info_to_save(data_dict = tmp_dict,
                                         name = "output_layer_tau2",
                                         neurons = self.output_layer_tau2.neurons)
        data["neuron params"] = tmp_dict

        tmp_dict = {}
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "input_layer_theta to liquid",
                                             src_layer = self.input_layer_theta,
                                             dst_layer = self.liquid_neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "input_layer_theta_dot to liquid",
                                             src_layer = self.input_layer_theta_dot,
                                             dst_layer = self.liquid_neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "liquid to readout_layer_tau1",
                                             src_layer = self.liquid_neurons,
                                             dst_layer = self.readout_layer_tau1)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "liquid to readout_layer_tau2",
                                             src_layer = self.liquid_neurons,
                                             dst_layer = self.readout_layer_tau2)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "readout_layer_tau1 to output_layer_tau1",
                                             src_layer = self.readout_layer_tau1,
                                             dst_layer = self.output_layer_tau1)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "readout_layer_tau2 to output_layer_tau2",
                                             src_layer = self.readout_layer_tau2,
                                             dst_layer = self.output_layer_tau2)
        self._append_internal_connection_info_to_save(data_dict = tmp_dict,
                                                      name = "liquid neurons",
                                                      liquid = self.liquid_neurons)
        data["connection params"] = tmp_dict


        fout = open(file_name, 'w')
        fout.write(yaml.dump(data, default_flow_style = False))
        fout.close()

    def load(self, file_name):

        fin = open(file_name, 'r')
        data = yaml.load(fin)
        fin.close()

        self._load_neurons(data, "input_layer_theta", self.input_layer_theta)
        self._load_neurons(data, "input_layer_theta_dot", self.input_layer_theta_dot)
        self._load_neurons(data, "liquid_neurons", self.liquid_neurons)
        self._load_neurons(data, "readout_layer_tau1", self.readout_layer_tau1)
        self._load_neurons(data, "readout_layer_tau2", self.readout_layer_tau2)
        self._load_neurons(data, "output_layer_tau1", self.output_layer_tau1)
        self._load_neurons(data, "output_layer_tau2", self.output_layer_tau2)

        self._load_connections(data, "input_layer_theta to liquid",
                               self.input_layer_theta, self.liquid_neurons)
        self._load_connections(data, "input_layer_theta_dot to liquid",
                               self.input_layer_theta_dot, self.liquid_neurons)
        self._load_connections_to_readout_layer(data, "liquid to readout_layer_tau1",
                                                self.liquid_neurons, self.readout_layer_tau1)
        self._load_connections_to_readout_layer(data, "liquid to readout_layer_tau2",
                                                self.liquid_neurons, self.readout_layer_tau2)
        self._load_connections(data, "readout_layer_tau1 to output_layer_tau1",
                               self.readout_layer_tau1, self.output_layer_tau1)
        self._load_connections(data, "readout_layer_tau2 to output_layer_tau2",
                               self.readout_layer_tau2, self.output_layer_tau2)
        self._load_internal_connections(data, "liquid neurons", self.liquid_neurons)        
        
    # layer is NeuronLayer type or LiquidNeurons type. 
    def _load_neurons(self, data, name, layer):

        tmp_data = data["neuron params"][name]

        layer.replace_neurons(tmp_data["size"], tmp_data["model"])

        # nest.SetStatusで書き換えられないキーを除外
        for itr in tmp_data["params"]:
            del itr["model"]
            del itr["element_type"]
            del itr["t_spike"]
            del itr["thread"]
            del itr["thread_local_id"]
            del itr["vp"]
            del itr["local_id"]
            del itr["local"]
            del itr["global_id"]
            del itr["archiver_length"]
            del itr["node_uses_wfr"]
            del itr["parent"]
            del itr["recordables"]
            del itr["supports_precise_spikes"]
            
        nest.SetStatus(layer.neurons, tmp_data["params"])

    # src_layer and dst_layer are NeuronLayer type or LiquidNeurons type. 
    def _load_connections(self, data, name, src_layer, dst_layer):

        tmp_data = data["connection params"][name]

        # 既存の結合はそこにはない前提
        for itr in tmp_data["params"]:
            tmp_param = copy.deepcopy(itr)
            del tmp_param["source"]
            del tmp_param["target"]
            del tmp_param["receptor"]
            tmp_param["model"] = tmp_param.pop("synapse_model") # renaming the key
            nest.Connect([src_layer.neurons[itr["source"]]],
                         [dst_layer.neurons[itr["target"]]],
                         syn_spec = tmp_param)

    # 通常の_load_connectionsに加え, readout layerのpresynaptic_neuronsの更新も必要
    def _load_connections_to_readout_layer(self, data, name, liquid, readout):

        self._load_connections(data, name, liquid, readout)

        conns = nest.GetConnections(liquid.neurons, readout.neurons)
        readout.presynaptic_neurons.clear()
        readout.connected_liquid = liquid

        for itr in conns:
            params = nest.GetStatus([itr])[0]
            ns = params["source"]
            nt = params["target"]
            if nt not in readout.presynaptic_neurons:
                readout.presynaptic_neurons[nt] = []
            readout.presynaptic_neurons[nt].append(ns)

            
    def _load_internal_connections(self, data, name, liquid):

        tmp_data = data["connection params"][name]

        # 既存の結合はそこにはない前提
        for itr in tmp_data["params"]:
            tmp_param = copy.deepcopy(itr)
            del tmp_param["source"]
            del tmp_param["target"]
            del tmp_param["receptor"]
            tmp_param["model"] = tmp_param.pop("synapse_model") # renaming the key
            nest.Connect([liquid.neurons[itr["source"]]],
                         [liquid.neurons[itr["target"]]],
                         syn_spec = tmp_param)
        
        
    # i_theta, i_theta_dot [pA]
    def simulate(self, sim_time, i_theta, i_theta_dot, filter_size):

        self.input_layer_theta.set_input_current(i_theta)
        self.input_layer_theta_dot.set_input_current(i_theta_dot)

        nest.Simulate(sim_time)

        ret1 = self.output_layer_tau1.get_mean_membrane_voltage(filter_size)
        ret2 = self.output_layer_tau2.get_mean_membrane_voltage(filter_size)
        
        return (ret1, ret2)

    # st: spike train (float list).
    # The dynamic state of the network will be reset.
    def train(self,
              i_theta,
              i_theta_dot,
              st_tau1_ref,
              st_tau2_ref,
              update_num = 100,
              sim_time = 1000.0,
              print_message = True):

        for i in range(update_num):

            if print_message:
                sys.stdout.write("training: " + str(i+1) + "/" + str(update_num) +  "    \r")
                sys.stdout.flush()
                
            nest.ResetNetwork()
            
            self.input_layer_theta.set_input_current(i_theta)
            self.input_layer_theta_dot.set_input_current(i_theta_dot)

            nest.Simulate(sim_time)
        
            self.readout_layer_tau1.train(st_tau1_ref)
            self.readout_layer_tau2.train(st_tau2_ref)

        if print_message:
            sys.stdout.write("\n")
            
        nest.ResetNetwork()        



    def _append_connection_info_to_save(self, data_dict, name, src_layer, dst_layer):

        data_dict[name] = {}

        source_ids = list(src_layer.neurons)
        target_ids = list(dst_layer.neurons)
        source_ids.sort()       # 一応
        target_ids.sort()       # 一応

        conns = nest.GetConnections(src_layer.neurons, dst_layer.neurons)
        
        # params
        data_dict[name]["params"] = list(nest.GetStatus(conns))
        
        for itr in data_dict[name]["params"]:
            tmp_str = itr["synapse_model"].name
            itr["synapse_model"] = tmp_str# pynestkernel.SLILiteral型をstr型にする
            
            # source, targetのidを各layerでのインデックスに
            itr["source"] = source_ids.index(itr["source"])
            itr["target"] = target_ids.index(itr["target"])
            
        # model
        used_models = list(set(nest.GetStatus(conns, "synapse_model"))) # 普通は要素数1 i.e. 単一のモデルのみ使用
        data_dict[name]["model"] = used_models[0].name if len(used_models) == 1 else "various models"
        
        # rule
        rows = len(source_ids)
        cols = len(target_ids)
        tmp_array = np.zeros((rows, cols)) # 各ニューロン間の接続の個数を格納する
        for itr in data_dict[name]["params"]:
            tmp_array[itr["source"], itr["target"]] += 1
        if np.allclose(tmp_array, np.ones((rows, cols))):
            data_dict[name]["rule"] = "all_to_all"
        elif rows == cols and np.allclose(tmp_array, np.eye(rows)):
            data_dict[name]["rule"] = "one_to_one"
        else:
            data_dict[name]["rule"] = "no rules found"


    def _append_internal_connection_info_to_save(self, data_dict, name, liquid):

        data_dict[name] = {}

        ids = list(liquid.neurons)
        ids.sort()       # 一応

        conns = nest.GetConnections(liquid.neurons, liquid.neurons)
        
        # params
        data_dict[name]["params"] = list(nest.GetStatus(conns))
        
        for itr in data_dict[name]["params"]:
            tmp_str = itr["synapse_model"].name
            itr["synapse_model"] = tmp_str# pynestkernel.SLILiteral型をstr型にする
            
            # source, targetのidを各layerでのインデックスに
            itr["source"] = ids.index(itr["source"])
            itr["target"] = ids.index(itr["target"])
            
        # model
        used_models = list(set(nest.GetStatus(conns, "synapse_model"))) # 普通は要素数1 i.e. 単一のモデルのみ使用
        data_dict[name]["model"] = used_models[0].name if len(used_models) == 1 else "various models"

        # rule
        data_dict[name]["rule"] = "no rules found"


        

    def _append_neuron_info_to_save(self, data_dict, name, neurons):

        data_dict[name] = {}

        # size
        data_dict[name]["size"] = len(neurons) # ニューロンの数

        # model
        used_models = list(set(nest.GetStatus(neurons, "model"))) # 普通は要素数1 i.e. 単一のモデルのみ使用
        data_dict[name]["model"] = used_models[0].name if len(used_models) == 1 else "various models"

        # params
        data_dict[name]["params"] = list(nest.GetStatus(neurons))
        for itr in data_dict[name]["params"]:# 各種pynestkernel.SLILiteral型をstr型にする
            tmp_str = itr["element_type"].name
            itr["element_type"] = tmp_str
            tmp_str = itr["model"].name
            itr["model"] = tmp_str
            tmp_list = []
            for itr_recordables in itr["recordables"]:
                tmp_list.append(itr_recordables.name)
            itr["recordables"] = tuple(tmp_list)



        

# coding: utf-8

import neuron_layer
import liquid_neurons
import nest
import numpy as np

import yaml

import sys

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
        
        self.liquid_neurons = liquid_neurons.LiquidNeurons(liquid_neurons_size, 0.005, 0.25)
        
        self.readout_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size)
        self.readout_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau2_size)

        self.output_layer_tau1 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        self.output_layer_tau2 = neuron_layer.NeuronLayer(readout_neurons_tau1_size,
                                                          V_th = float(10**100))
        
        # connect layers
        self.input_layer_theta.connect2liquid(self.liquid_neurons, 0.3, 0.25)
        self.input_layer_theta_dot.connect2liquid(self.liquid_neurons, 0.3, 0.25)
        self.liquid_neurons.connect(self.readout_layer_tau1, 0.3, 0.25)
        self.liquid_neurons.connect(self.readout_layer_tau2, 0.3, 0.25)
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
                                             sources = self.input_layer_theta.neurons,
                                             targets = self.liquid_neurons.neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "input_layer_theta_dot to liquid",
                                             sources = self.input_layer_theta_dot.neurons,
                                             targets = self.liquid_neurons.neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "liquid to readout_layer_tau1",
                                             sources = self.liquid_neurons.neurons,
                                             targets = self.readout_layer_tau1.neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "liquid to readout_layer_tau2",
                                             sources = self.liquid_neurons.neurons,
                                             targets = self.readout_layer_tau2.neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "readout_layer_tau1 to output_layer_tau1",
                                             sources = self.readout_layer_tau1.neurons,
                                             targets = self.output_layer_tau1.neurons)
        self._append_connection_info_to_save(data_dict = tmp_dict,
                                             name = "readout_layer_tau2 to output_layer_tau2",
                                             sources = self.readout_layer_tau2.neurons,
                                             targets = self.output_layer_tau2.neurons)
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

        # self._load_connections(data, "input_layer_theta to liquid",
        #                        self.input_layer_theta, self.liquid_neurons)
        # self._load_connections(data, "input_layer_theta_dot to liquid",
        #                        self.input_layer_theta_dot, self.liquid_neurons)
        # self._load_connections(data, "liquid to readout_layer_tau1",
        #                        self.liquid_neurons, self.readout_layer_tau1)
        # self._load_connections(data, "liquid to readout_layer_tau2",
        #                        self.liquid_neurons, self.readout_layer_tau2)
        # self._load_connections(data, "readout_layer_tau1 to output_layer_tau1",
        #                        self.readout_layer_tau1, self.output_layer_tau1)
        # self._load_connections(data, "readout_layer_tau2 to output_layer_tau2",
        #                        self.readout_layer_tau2, self.output_layer_tau2)
        
        
    # layer is NeuronLayer type or LiquidNeurons type. 
    def _load_neurons(self, data, name, layer):

        tmp_data = data["neuron params"][name]

        if tmp_data["size"] != len(layer.neurons) or tmp_data["model"] != layer.neuron_model:
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

    # # src_layer and dst_layer are NeuronLayer type or LiquidNeurons type. 
    # def _load_connections(self, data, name, src_layer, dst_layer):

    #     tmp_data = data["connection params"][name]

        
        
        
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



    def _append_connection_info_to_save(self, data_dict, name, sources, targets):

        data_dict[name] = {}

        # params
        data_dict[name]["params"] = list(nest.GetStatus(nest.GetConnections(sources, targets)))
        for itr in data_dict[name]["params"]:# "synapse_model" のpynestkernel.SLILiteral型をstr型にする
            tmp_str = itr["synapse_model"].name
            itr["synapse_model"] = tmp_str

        
        # model
        used_models = list(set(nest.GetStatus(nest.GetConnections(sources, targets),
                                              "synapse_model"))) # 普通は要素数1 i.e. 単一のモデルのみ使用
        data_dict[name]["model"] = used_models[0].name if len(used_models) == 1 else "various models"

        # rule
        source_ids_set = set()
        target_ids_set = set()
        for itr in data_dict[name]["params"]:
            source_ids_set.add(itr["source"])
            target_ids_set.add(itr["target"])
        source_ids = list(source_ids_set)
        target_ids = list(target_ids_set)
        source_ids.sort()       # 全種類のsourcesのidを昇順に並べたもの
        target_ids.sort()       # 全種類のtargetsのidを昇順に並べたもの
        rows = len(sources)
        cols = len(targets)
        tmp_array = np.zeros((rows, cols)) # 各ニューロン間の接続の個数を格納する
        for itr in data_dict[name]["params"]:
            tmp_array[source_ids.index(itr["source"]), target_ids.index(itr["target"])] += 1
        if np.allclose(tmp_array, np.ones((rows, cols))):
            data_dict[name]["rule"] = "all_to_all"
        elif rows == cols and np.allclose(tmp_array, np.eye(rows)):
            data_dict[name]["rule"] = "one_to_one"
        else:
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



        

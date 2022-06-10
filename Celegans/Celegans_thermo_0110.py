#!/usr/bin/env python
# coding: utf-8

import os
#
# os.chdir("../../")
# print(os.path.abspath('.'))

import spaic
import torch
# from spaic.Learning.STCA_Learner import STCA
from spaic.Learning.Learner import Learner
from spaic import Learning
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

run_time = 200
# run_time = 100
bat_size = 1
# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
backend = spaic.Torch_Backend(device)
sim_name = backend.backend_name
sim_name = sim_name.lower()
backend.dt = 0.1
# 创建训练数据集
# root = './spaic/Datasets/MNIST'
# test_set = spaic.MNIST(root, is_train=False)
# myset = spaic.CustomDataset(data=[0.1, 1, 20, 3, 0.1], label=[1, 1, 1, 1, 1])
# data = [10 + i/100 for i in range(1, 100)]
# # data = []
# data.append(20)
# data.append(1)
# label = []
# import random
# for i in range(len(data)):
#     label.append(random.randint(0, 10))
#mdata = 9
#mdata = 0.1

mdata = 0
myset = spaic.CustomDataset(data=[[mdata]], label=[1])
# myset = spaic.CustomDataset(data=[[0.02]], label=[1])
# 创建DataLoader迭代器
# dataloader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)
dataloader = spaic.Dataloader(myset, batch_size=1, shuffle=False)

'''
name_list = ['RE', 'LE', 'RI', 'LI']
In_W = 100.0
In_std = 5.0
W = {"E": 90.8, "I": -120.5}
std = {"E": 5., "I": 5.}
'''

seed = 2019

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

wmax = {"E": 100000.0, "I": 0}
wmin = {"E": 0, "I": -100000.0}
delay_E = 0.0
delay_I = 0.0
delay = {"E": delay_E, "I": delay_I}
segment = 10

from spaic.Neuron.Node import Generator

class Thermo_Generator(Generator):
    """
        恒定电流生成器。
        Generate a constant current input.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Thermo_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num

    def torch_coding(self, source, device):
        assert (source >= 0).all(), "Input rate must be non-negative"
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)

        if source.ndim == 0:
            batch = 1
        else:
            batch = source.shape[0]

        shape = [batch, self.num]
        spk_shape = [self.time_step] + list(shape)
        # spikes = source * np.sin(torch.ones(spk_shape, device=device))
        spikes = source * torch.ones(spk_shape, device=device)
        # spikes = source * torch.sin(spk_shape, device=device)
        for i in range(len(spikes)):
            spikes[i][0][0] =20*torch.sin(torch.tensor(i/50))
        return spikes

Generator.register('thermo_generator', Thermo_Generator)
class Thermo2_Generator(Generator):
    """
        恒定电流生成器。
        Generate a constant current input.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Thermo2_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num

    def torch_coding(self, source, device):
        assert (source >= 0).all(), "Input rate must be non-negative"
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)

        if source.ndim == 0:
            batch = 1
        else:
            batch = source.shape[0]

        shape = [batch, self.num]
        spk_shape = [self.time_step] + list(shape)
        # spikes = source * np.sin(torch.ones(spk_shape, device=device))
        spikes = source * torch.ones(spk_shape, device=device)
        # spikes = source * torch.sin(spk_shape, device=device)
        for i in range(len(spikes)):
            spikes[i][0][0] =torch.cos(torch.tensor(i/50))
        return spikes

Generator.register('thermo2_generator', Thermo2_Generator)


class Thermo3_Generator(Generator):
    """
        恒定电流生成器。
        Generate a constant current input.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Thermo3_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num

    def torch_coding(self, source, device):
        assert (source >= 0).all(), "Input rate must be non-negative"
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)

        if source.ndim == 0:
            batch = 1
        else:
            batch = source.shape[0]

        shape = [batch, self.num]
        spk_shape = [self.time_step] + list(shape)
        # spikes = source * np.sin(torch.ones(spk_shape, device=device))
        spikes = source * torch.ones(spk_shape, device=device)
        # spikes = source * torch.sin(spk_shape, device=device)
        for i in range(len(spikes)):
            spikes[i][0][0] =-torch.cos(torch.tensor(i/50))
        return spikes

Generator.register('thermo3_generator', Thermo3_Generator)
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        # coding
        # self.input = spaic.Generator(num=38, coding_method='gaussian_generator')
        #self.input1 = spaic.Generator(num=1, coding_method='cc_generator')
        self.input1 = spaic.Generator(num=1, coding_method='thermo_generator')
        self.input4 = spaic.Generator(num=1, coding_method='thermo2_generator')
        self.input7 = spaic.Generator(num=1, coding_method='thermo3_generator')

        self.bias2 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias3 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias4 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias5 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias7 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias8 = spaic.Generator(num=1, coding_method='cc_generator')
        self.bias10 = spaic.Generator(num=1, coding_method='cc_generator')

        # self.connectionIn_AVB = spaic.Connection(self.input, self.AVB, link_type='full',
        #                                            w_mean=40.0)

        # self.AVBL = spaic.NeuronGroup(1, neuron_model='aeif')

        for i in range(1, 10):
            self.add_assembly(name="N" + str(i),
                              assembly=spaic.NeuronGroup(1, neuron_model='aeif', tau_w=3, tau_m=2, a=2, b=0,
                                                           delta_t=10, delta_t2=2))
        self.add_assembly(name="N10",
                              assembly=spaic.NeuronGroup(1, neuron_model='aeif', tau_w=6, tau_m=2, a=2, b=0,
                                                           delta_t=10, delta_t2=2))

        self.connectionIn1_N1 = spaic.Connection(self.input1, self.N1, link_type='full',
                                                   w_mean=1.50, w_std=0.)
        # self.connectionIn4_N4 = spaic.Connection(self.input4, self.N4, link_type='full',
        #                                            w_mean=6, w_std=0.)
        # self.connectionIn7_N7 = spaic.Connection(self.input7, self.N7, link_type='full',
        #                                            w_mean=6, w_std=0.)
        self.connectionbias2_N = spaic.Connection(self.bias2, self.N2, link_type='full',
                                                    w_mean=15, w_std=0.)
        self.connectionbias3_N = spaic.Connection(self.bias3, self.N3, link_type='full',
                                                    w_mean=-15, w_std=0.)
        self.connectionbias4_N = spaic.Connection(self.bias4, self.N4, link_type='full',
                                                    w_mean=25, w_std=0.)
        self.connectionbias5_N = spaic.Connection(self.bias5, self.N5, link_type='full',
                                                    w_mean=25, w_std=0)
        self.connectionbias7_N = spaic.Connection(self.bias7, self.N7, link_type='full',
                                                    w_mean=15, w_std=0.)
        self.connectionbias8_N = spaic.Connection(self.bias8, self.N8, link_type='full',
                                                    w_mean=15, w_std=0.)
        self.connectionbias10_N = spaic.Connection(self.bias10, self.N10, link_type='full',
                                                     w_mean=5, w_std=0.)

        #         self.connectionAVB_DB = spaic.Connection(self.AVB, self.DB1, link_type='full',
        #                                                    synapse=True,
        #                                                    synapse_type='electrical_synapse',
        #                                                    pre_var_name='V',
        #                                                    w_mean=1.0)
        #         self.connectionAVB_VB = spaic.Connection(self.AVB, self.VB1, link_type='full',
        #                                                    synapse=True,
        #                                                    synapse_type='electrical_synapse',
        #                                                    pre_var_name='V',
        #                                                    w_mean=1.0)

        import pandas as pd
        file_name = 'CElegans_thermo.xls'

        df = pd.read_excel(file_name, header=0, sheet_name='ThermoCircuit')
        print("df", df)

        name_list = self._groups.keys()
        print("name_list", name_list)

        for i in range(df.__len__()):
            data = df.iloc[i]
            if (data['Origin'] in name_list) and (data['Target'] in name_list):
                conn_name = "connection" + data['Origin'] + '_' + data['Target']
                pre = self._groups[data['Origin']]
                post = self._groups[data['Target']]
                if data['Neurotransmitter'] == 'Acetylcholine':
                    self.add_connection(name=conn_name + 'in',
                                        connection=spaic.Connection(
                                            pre_assembly=pre,
                                            post_assembly=post,
                                            link_type='full',
                                            w_mean=data['Weight'],
                                            w_std=0.0
                                        ))
                else:
                    self.add_connection(name=conn_name + 'in',
                                        connection=spaic.Connection(
                                            pre_assembly=pre,
                                            post_assembly=post,
                                            link_type='full',
                                            w_mean=data['Weight'],
                                            w_std=0.0
                                        ))

        # self.learner1 = spaic.Learner(trainable=[self.connectionN5_N6in],
        #                                     algorithm='celegans_thermo_gradient')
        # self.learner2 = spaic.Learner(trainable=[self.connectionN8_N9in],
        #                                 algorithm='celegans_thermo_gradient')

        # df_muscle = pd.read_excel(file_name, header=False, sheet_name='NeuronsToMuscle')
        # for data in df_muscle:
        #     if (data['Neuron'] in name_list) and (data['Muscle'] in name_list):
        #         conn_name = "connection" + data['Neuron'] + '_' + data['Muscle']
        #         pre = self._groups[data['Neuron']]
        #         post = self._groups[data['Muscle']]
        #         if data['Neurotransmitter'] == 'GABA':
        #             self.add_connection(name=conn_name + 'in',
        #                                 connection=spaic.Connection(
        #                                     pre_assembly=pre,
        #                                     post_assembly=post,
        #                                     link_type='full',
        #                                     w_mean=-data['Weight'],
        #                                 ))
        #         else:
        #             self.add_connection(name=conn_name + 'ex',
        #                                 connection=spaic.Connection(
        #                                     pre_assembly=pre,
        #                                     post_assembly=post,
        #                                     link_type='full',
        #                                     w_mean=-data['Weight'],
        #                                 ))

        # Monitor
        #         self.mon_V = spaic.StateMonitor(self.N1, 'V')
        #         self.mon_V2 = spaic.StateMonitor(self.N1, 'WgtSum')
        #         self.mon_V = spaic.StateMonitor(self.N6, 'V')
        #         self.mon_V2 = spaic.StateMonitor(self.N6, 'WgtSum')
        #         self.mon_V = spaic.StateMonitor(self.N9, 'V')
        #         self.mon_V2 = spaic.StateMonitor(self.N9, 'WgtSum')
        self.mon_V = spaic.StateMonitor(self.input1, 'O')
        self.mon_V4 = spaic.StateMonitor(self.input4, 'O')
        self.mon_V7 = spaic.StateMonitor(self.input7, 'O')
        self.mon_V2 = spaic.StateMonitor(self.N10, 'V')
        self.mon_N1_O = spaic.StateMonitor(self.N1, 'O')
        self.mon_N4_O = spaic.StateMonitor(self.N4, 'O')
        self.mon_N5_O = spaic.StateMonitor(self.N5, 'O')
        self.mon_N6_O = spaic.StateMonitor(self.N6, 'O')
        self.mon_N7_O = spaic.StateMonitor(self.N7, 'O')
        self.mon_N8_O = spaic.StateMonitor(self.N8, 'O')
        self.mon_N9_O = spaic.StateMonitor(self.N9, 'O')
        self.mon_N10_O = spaic.StateMonitor(self.N10, 'O')
        self.mon_I = spaic.SpikeMonitor(self.input1, 'O')
        # self.mon_I = spaic.SpikeMonitor(self.input4, 'O')
        # self.mon_I = spaic.SpikeMonitor(self.input7, 'O')
        self.mon_N1 = spaic.StateMonitor(self.N1, 'V')
        self.mon_N2 = spaic.StateMonitor(self.N2, 'V')
        self.mon_N4 = spaic.StateMonitor(self.N4, 'V')
        self.mon_N5 = spaic.StateMonitor(self.N5, 'V')
        self.mon_N7 = spaic.StateMonitor(self.N7, 'V')
        self.mon_N8 = spaic.StateMonitor(self.N8, 'V')
        self.mon_N6 = spaic.StateMonitor(self.N6, 'V')
        self.mon_N9 = spaic.StateMonitor(self.N9, 'V')
        self.mon_N10 = spaic.StateMonitor(self.N10, 'V')
        # self.mon_IN = spaic.SpikeMonitor(self.noise_input, 'O')
        self.spk_N1 = spaic.SpikeMonitor(self.N1, 'O')
        self.spk_N2 = spaic.SpikeMonitor(self.N2, 'O')
        self.spk_N3 = spaic.SpikeMonitor(self.N3, 'O')
        self.spk_N4 = spaic.SpikeMonitor(self.N4, 'O')
        self.spk_N5 = spaic.SpikeMonitor(self.N5, 'O')
        self.spk_N6 = spaic.SpikeMonitor(self.N6, 'O')
        self.spk_N7 = spaic.SpikeMonitor(self.N7, 'O')
        self.spk_N8 = spaic.SpikeMonitor(self.N8, 'O')
        self.spk_N9 = spaic.SpikeMonitor(self.N9, 'O')
        self.spk_N10 = spaic.SpikeMonitor(self.N10, 'O')

        #
        # self.spk_DB7 = spaic.SpikeMonitor(self.RE7, 'O')
        # self.spk_DD7 = spaic.SpikeMonitor(self.RI7, 'O')
        # self.spk_VB7 = spaic.SpikeMonitor(self.LE7, 'O')
        # self.spk_VD7 = spaic.SpikeMonitor(self.LI7, 'O')
        #
        # self.spk_DB8 = spaic.SpikeMonitor(self.RE8, 'O')
        # self.spk_DD8 = spaic.SpikeMonitor(self.RI8, 'O')
        # self.spk_VB8 = spaic.SpikeMonitor(self.LE8, 'O')
        # self.spk_VD8 = spaic.SpikeMonitor(self.LI8, 'O')
        #
        # self.spk_DB9 = spaic.SpikeMonitor(self.RE9, 'O')`
        # self.spk_DD9 = spaic.SpikeMonitor(self.RI9, 'O')
        # self.spk_VB9 = spaic.SpikeMonitor(self.LE9, 'O')
        # self.spk_VD9 = spaic.SpikeMonitor(self.LI9, 'O')
        #
        # self.spk_DB10 = spaic.SpikeMonitor(self.RE10, 'O')
        # self.spk_DD10 = spaic.SpikeMonitor(self.RI10, 'O')
        # self.spk_VB10 = spaic.SpikeMonitor(self.LE10, 'O')
        # self.spk_VD10 = spaic.SpikeMonitor(self.LI10, 'O')

        # self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        # self.learner.set_optimizer('Adam', 0.001)
        self.set_backend(backend)


Net = TestNet()
Net.build()
print(Net)

import matplotlib.pyplot as plt
print("Start running")
end = run_time
print("Start running")

#for simulate in range(3):
for i, item in enumerate(dataloader):
    data, label = item
    Net.input1(data)
    Net.input4(data)
    Net.input7(data)
    Net.bias2(0.8305)
    Net.bias3(0.396)
    Net.bias4(0.6)
    Net.bias5(0.7)
    Net.bias7(0.6)
    Net.bias8(1)
    Net.bias10(0.205)
    # Net.noise_input(data)
    Net.run(run_time)
    print("Net.mon_V.values",Net.mon_V.values[0][0])

    # time_line = Net.mon_IR
    # output_line = Net.mon_IR

    m = dict()
    m = {'mN_i': [],
         'mN_t': [],
         }
    for i in range(1, segment + 1):
        for key in ['N']:
            dict_i = 'm' + key + '_i'
            dict_t = 'm' + key + '_t'
            monit = 'spk_' + key + str(i)
            m[dict_i].append(Net._monitors[monit].spk_index[0])
            m[dict_t].append(Net._monitors[monit].spk_times[0])

    ax1=plt.subplot(5, 1, 1)
    # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # plt.plot(output_line, '.', label='input spike')
    plt.plot(Net.mon_V.times, Net.mon_V.values[0][0])
    # plt.plot(Net.mon_V.times, Net.mon_N1_O.values[0][0]*50 - 100, '.')
    plt.xlim([-1, end + 1])
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.ylabel("N1 Input",fontdict={'family':'Times New Roman', 'size':20})
    # plt.plot(time_line, vth_line, label='Threshold')
    # plt.xlim([-1, end + 1])
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.ylim([-100, 0])
    plt.legend()

    # plt.subplot(9, 1, 2)
    # # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # # plt.plot(output_line, '.', label='input spike')
    # plt.plot(Net.mon_V.times, Net.mon_V4.values[0][0])
    # # plt.plot(Net.mon_V.times, Net.mon_N1_O.values[0][0]*50 - 100, '.')
    # plt.ylabel("N4 Input")
    # # plt.plot(time_line, vth_line, label='Threshold')
    # plt.xlim([-1, end + 1])
    # # plt.ylim([-100, 0])
    # plt.legend()
    #
    # plt.subplot(9, 1, 3)
    # # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # # plt.plot(output_line, '.', label='input spike')
    # plt.plot(Net.mon_V.times, Net.mon_V7.values[0][0])
    # # plt.plot(Net.mon_V.times, Net.mon_N1_O.values[0][0]*50 - 100, '.')
    # plt.ylabel("N7 Input")
    # # plt.plot(time_line, vth_line, label='Threshold')
    # plt.xlim([-1, end + 1])
    # # plt.ylim([-100, 0])
    # plt.legend()

    ax2=plt.subplot(5, 1, 2)
    # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # plt.plot(output_line, '.', label='input spike')
    plt.plot(Net.mon_V.times, Net.mon_N10.values[0][0])
    plt.plot(Net.mon_V.times, Net.mon_N10_O.values[0][0] * 50 - 100, '.')
    # plt.plot(Net.mon_IN.spk_times[0], Net.mon_IN.spk_index[0], '.')
    plt.xlim([-1, end + 1])
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.ylabel("N10",fontdict={'family':'Times New Roman', 'size':20})
    # plt.plot(time_line, vth_line, label='Threshold')
    plt.ylim([-100, 0])
    #plt.xlim([-1, end + 1])
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.legend()

    #print(segment)

    # plt.subplot(5, 1, 3)
    # # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # # plt.plot(output_line, '.', label='input spike')
    # plt.plot(Net.mon_V.times, Net.mon_N7.values[0][0])
    # plt.plot(Net.mon_V.times, Net.mon_N7_O.values[0][0] * 50 - 100, '.')
    # # plt.plot(Net.mon_IN.spk_times[0], Net.mon_IN.spk_index[0], '.')
    # plt.ylabel("N7")
    # # plt.plot(time_line, vth_line, label='Threshold')
    # plt.ylim([-100, 0])
    # plt.xlim([-1, end + 1])
    # plt.legend()

    # plt.subplot(9, 1, 6)
    # # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # # plt.plot(output_line, '.', label='input spike')
    # plt.plot(Net.mon_V.times, Net.mon_N5.values[0][0])
    # plt.plot(Net.mon_V.times, Net.mon_N5_O.values[0][0] * 50 - 100, '.')
    # # plt.plot(Net.mon_IN.spk_times[0], Net.mon_IN.spk_index[0], '.')
    # plt.ylabel("N5")
    # # plt.plot(time_line, vth_line, label='Threshold')
    # plt.ylim([-100, 0])
    # plt.xlim([-1, end + 1])
    # plt.legend()

    ax3=plt.subplot(5, 1, 3)
    # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # plt.plot(output_line, '.', label='input spike')
    plt.plot(Net.mon_V.times, Net.mon_N6.values[0][0])
    plt.plot(Net.mon_V.times, Net.mon_N6_O.values[0][0] * 50 - 100, '.')
    # plt.plot(Net.mon_IN.spk_times[0], Net.mon_IN.spk_index[0], '.')
    plt.xlim([-1, end + 1])
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.ylabel("N6",fontdict={'family':'Times New Roman', 'size':20})
    # plt.plot(time_line, vth_line, label='Threshold')
    plt.ylim([-100, 0])
    #plt.xlim([-1, end + 1])
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.legend()

    ax4=plt.subplot(5, 1, 4)
    # plt.plot(time_line, np.mean(Net.spk_RE1.time_spk_rate, axis=0), label='V')
    # plt.plot(output_line, '.', label='input spike')
    plt.plot(Net.mon_V.times, Net.mon_N9.values[0][0])
    plt.plot(Net.mon_V.times, Net.mon_N9_O.values[0][0] * 50 - 100, '.')
    # plt.plot(Net.mon_IN.spk_times[0], Net.mon_IN.spk_index[0], '.')
    plt.xlim([-1, end + 1])
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.ylabel("N9",fontdict={'family':'Times New Roman', 'size':20})
    # plt.plot(time_line, vth_line, label='Threshold')
    plt.ylim([-100, 0])
    #plt.xlim([-1, end + 1])
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.legend()

    ax5=plt.subplot(5, 1, 5)
    for i in range(segment):
        plt.plot(m['mN_t'][i], m['mN_i'][i] + i, '.')
    # plt.ylim((-0.1, 2.5))
    #plt.xlim([-1, end + 1])
    plt.xlim([-1, end + 1])
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.ylabel("spike time",fontdict={'family':'Times New Roman', 'size':20})
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.legend()

    plt.xlim([-1, end + 1])
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.legend()

    plt.subplots_adjust(left=0.11,
                        bottom=0.11,
                        right=0.9,
                        top=0.88,
                        wspace=0.2,
                        hspace=0.2)

    # plt.savefig('/home/naturia/Desktop/thermo_spike3.png', dpi=800)

    plt.show()

    # import seaborn as sns

    # print("h")
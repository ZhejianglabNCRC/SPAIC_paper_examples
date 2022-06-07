# -*- coding: utf-8 -*-
"""
Created on 2021/8/5
@project: main.py
@filename: CorticalColumnTest
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""

import spaic
import torch

from tqdm import tqdm
from spaic.IO.Dataset import MNIST as dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process, Lock
from matplotlib.animation import FuncAnimation
from time import time


# 设备设置
if torch.cuda.is_available():
    device = 'cuda'

else:
    device = 'cpu'
print(device)
e_w = 0.02
i_w = -0.08

class Layer(spaic.Assembly):

    def __init__(self, exc_num, inh_num, ee_p, ii_p, ei_p, ie_p):
        super(Layer, self).__init__()
        self.excitory_neuron = spaic.NeuronGroup(neuron_number=exc_num, neuron_type='exc', neuron_model='lif', tau_m=20.0)
        self.inhibitory_neuron = spaic.NeuronGroup(neuron_number=inh_num, neuron_type='inh', neuron_model='lif', tau_m=10.0)

        self.exc_possion = spaic.Generator(num=exc_num, dec_target=self.excitory_neuron, coding_method='poisson_generator',
                                             coding_var_name='Isyn', rate=0.01)
        self.inh_possion = spaic.Generator(num=inh_num, dec_target=self.inhibitory_neuron, coding_method='poisson_generator',
                                             coding_var_name='Isyn', rate=0.0)
        self.ee_connection = spaic.Connection(pre_assembly=self.excitory_neuron, post_assembly=self.excitory_neuron,
                                                link_type='sparse_connection',  density=ee_p, w_min=0, w_mean=e_w)
        self.ei_connection = spaic.Connection(pre_assembly=self.excitory_neuron, post_assembly=self.inhibitory_neuron,
                                                link_type='sparse_connection',  density=ei_p, w_min=0, w_mean=e_w)
        self.ie_connection = spaic.Connection(pre_assembly=self.inhibitory_neuron, post_assembly=self.excitory_neuron,
                                                link_type='sparse_connection',  density=ie_p, w_max=0, w_mean=i_w)
        self.ii_connection = spaic.Connection(pre_assembly=self.inhibitory_neuron, post_assembly=self.inhibitory_neuron,
                                                link_type='sparse_connection',  density=ii_p, w_max=0, w_mean=i_w)

class LayerProj(spaic.Projection):

    def __init__(self, pre : Layer, post: Layer, ee_p, ei_p, ie_p, ii_p):
        super(LayerProj, self).__init__(pre, post)
        if ee_p > 0:
            self.ee_connection = spaic.Connection(pre_assembly=pre.excitory_neuron, post_assembly=post.excitory_neuron,
                                                  link_type='sparse_connection', density=ee_p, w_min=0, w_mean=e_w)
        if ei_p > 0:
            self.ei_connection = spaic.Connection(pre_assembly=pre.excitory_neuron, post_assembly=post.inhibitory_neuron,
                                                  link_type='sparse_connection', density=ei_p, w_min=0, w_mean=e_w)
        if ie_p > 0:
            self.ie_connection = spaic.Connection(pre_assembly=pre.excitory_neuron, post_assembly=post.inhibitory_neuron,
                                                  link_type='sparse_connection', density=ie_p, w_max=0, w_mean=i_w)
        if ii_p > 0:
            self.ii_connection = spaic.Connection(pre_assembly=pre.excitory_neuron, post_assembly=post.inhibitory_neuron,
                                                  link_type='sparse_connection', density=ii_p, w_max=0, w_mean=i_w)

class CorticalMicrocircuit(spaic.Network):

    def __init__(self):
        super(CorticalMicrocircuit, self).__init__()
        self.layer23 = Layer(exc_num=2068, inh_num=583, ee_p=0.101, ei_p=0.125, ii_p=0.289, ie_p=0.227)
        self.layer4 = Layer(exc_num=2191, inh_num=548, ee_p=0.05, ei_p=0.079, ii_p=0.26, ie_p=0.175)
        self.layer5 = Layer(exc_num=485, inh_num=106, ee_p=0.083, ei_p=0.06, ii_p=0.416, ie_p=0.323)
        self.layer6 = Layer(exc_num=1439, inh_num=395, ee_p=0.07, ei_p=0.066, ii_p=0.544, ie_p=0.385)

        self.proj23_4 = LayerProj(pre=self.layer23, post=self.layer4, ee_p=0.008, ei_p=0.069, ie_p=0.0, ii_p=0)
        self.proj23_5 = LayerProj(pre=self.layer23, post=self.layer5, ee_p=0.01, ei_p=0.0095, ie_p=0.062, ii_p=0.027)
        self.proj4_23 = LayerProj(pre=self.layer4, post=self.layer23, ee_p=0.024, ei_p=0.032, ie_p=0.01, ii_p=0)
        self.proj4_5 = LayerProj(pre=self.layer4, post=self.layer5, ee_p=0.051, ei_p=0.0026, ie_p=0.02, ii_p=0)
        self.proj4_6 = LayerProj(pre=self.layer4, post=self.layer6, ee_p=0.031, ei_p=0.0013, ie_p=0.01, ii_p=0)
        self.proj5_6 = LayerProj(pre=self.layer5, post=self.layer6, ee_p=0.0037, ei_p=0.002, ie_p=0.007, ii_p=0.0)
        self.proj6_5 = LayerProj(pre=self.layer6, post=self.layer5, ee_p=0.002, ei_p=0.001, ie_p=0.001, ii_p=0)

        self.layer23_v = spaic.StateMonitor(self.layer23.excitory_neuron, var_name='Isyn')

        self.l23exc_spk_monitor = spaic.SpikeMonitor(self.layer23.excitory_neuron)
        self.l23inh_spk_monitor = spaic.SpikeMonitor(self.layer23.inhibitory_neuron)
        self.l4exc_spk_monitor = spaic.SpikeMonitor(self.layer4.excitory_neuron)
        self.l4inh_spk_monitor = spaic.SpikeMonitor(self.layer4.inhibitory_neuron)
        self.l5exc_spk_monitor = spaic.SpikeMonitor(self.layer5.excitory_neuron)
        self.l5inh_spk_monitor = spaic.SpikeMonitor(self.layer5.inhibitory_neuron)
        self.l6exc_spk_monitor = spaic.SpikeMonitor(self.layer6.excitory_neuron)
        self.l6inh_spk_monitor = spaic.SpikeMonitor(self.layer6.inhibitory_neuron)


TestNet = CorticalMicrocircuit()




TestNet.set_backend('pytorch',device)
with torch.no_grad():
    t1 = time()
    TestNet.run(100.0)
    t2 = time()
    print(t2-t1)



# tim = TestNet.layer23_v.times
# vv = TestNet.layer23_v.values[0,0]
# plt.plot(tim, vv)
# plt.show()

l23e_num = 2068
l23i_num = 583
l4e_num = 2191
l4i_num = 548
l5e_num = 485
l5i_num = 106
l6e_num = 1439
l6i_num = 295
l23e_spk_tim = TestNet.l23exc_spk_monitor.spk_times[0]
l23e_spk_ind = TestNet.l23exc_spk_monitor.spk_index[0]
l23i_spk_tim = TestNet.l23inh_spk_monitor.spk_times[0]
l23i_spk_ind = TestNet.l23inh_spk_monitor.spk_index[0]

l4e_spk_tim = TestNet.l4exc_spk_monitor.spk_times[0]
l4e_spk_ind = TestNet.l4exc_spk_monitor.spk_index[0]
l4i_spk_tim = TestNet.l4inh_spk_monitor.spk_times[0]
l4i_spk_ind = TestNet.l4inh_spk_monitor.spk_index[0]

l5e_spk_tim = TestNet.l5exc_spk_monitor.spk_times[0]
l5e_spk_ind = TestNet.l5exc_spk_monitor.spk_index[0]
l5i_spk_tim = TestNet.l5inh_spk_monitor.spk_times[0]
l5i_spk_ind = TestNet.l5inh_spk_monitor.spk_index[0]


l6e_spk_tim = TestNet.l6exc_spk_monitor.spk_times[0]
l6e_spk_ind = TestNet.l6exc_spk_monitor.spk_index[0]
l6i_spk_tim = TestNet.l6inh_spk_monitor.spk_times[0]
l6i_spk_ind = TestNet.l6inh_spk_monitor.spk_index[0]

plt.plot(l23e_spk_tim, l23e_spk_ind, '.')
plt.plot(l23i_spk_tim, l23i_spk_ind+l23e_num , '.')
plt.plot(l4e_spk_tim, l4e_spk_ind+l23e_num+l23i_num, '.')
plt.plot(l4i_spk_tim, l4i_spk_ind+l23e_num+l23i_num+l4e_num, '.')
plt.plot(l5e_spk_tim, l5e_spk_ind+l23e_num+l23i_num+l4e_num+l4i_num, '.')
plt.plot(l5i_spk_tim, l5i_spk_ind+l23e_num+l23i_num+l4e_num+l4i_num+l5e_num, '.')
plt.plot(l6e_spk_tim, l6e_spk_ind+l23e_num+l23i_num+l4e_num+l4i_num+l5e_num+l5i_num, '.')
plt.plot(l6i_spk_tim, l6i_spk_ind+l23e_num+l23i_num+l4e_num+l4i_num+l5e_num+l5i_num+l6e_num, '.')
plt.show()






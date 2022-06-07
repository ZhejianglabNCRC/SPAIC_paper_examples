import torch
import torch.nn as nn
import numpy as np
import os
os.chdir("../../")
import spaic

class ActorNetSpiking_spaic(spaic.Network):
    def __init__(self,state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        super(ActorNetSpiking_spaic, self).__init__()

        # frontend setting
        # coding

        self.input = spaic.Encoder(num=state_num, coding_method='poisson',unit_conversion=1)

        # neuron group
        self.layer1 = spaic.NeuronGroup(hidden1, neuron_model='clif',v_th=1)
        self.layer2 = spaic.NeuronGroup(hidden2, neuron_model='clif',v_th=1)
        self.layer3 = spaic.NeuronGroup(hidden3, neuron_model='clif',v_th=1)
        self.layer4 = spaic.NeuronGroup(action_num, neuron_model='clif',v_th=1)
        # decoding
        self.output = spaic.Decoder(num=action_num, dec_target=self.layer4, coding_method='spike_counts')

        # w_mean=0.08 # 50 steps
        # w_std=0.02

        w_mean = 0.03
        w_std = 0.005

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection4 = spaic.Connection(self.layer3, self.layer4, link_type='full',w_mean=w_mean, w_std=w_std)
        # Learner
        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        # self.learner = spaic.Learner(trainable=self, algorithm='STBP')
        self.learner.set_optimizer('Adam', 0.001)

if __name__ == '__main__':
    actor_net=ActorNetSpiking_spaic(24,2,device='cuda')

    actor_net.set_simulator('torch', 'cpu')

    # for i in range(1):
    state_spikes=np.ones((1,24),dtype=float)*0.5
    actor_net.input(state_spikes)
    actor_net.run(5)
    action = actor_net.output.predict

    # state_spikes = np.ones((1, 24), dtype=float)
    # print(state_spikes.shape[0])
    # x=10
    # y=20
    # print('done{}'.format(x))
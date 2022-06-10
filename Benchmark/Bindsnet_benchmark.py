import torch
import torch.nn
from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.encoding import poisson
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input

torch.set_default_tensor_type("torch.FloatTensor")
# torch.set_default_tensor_type("torch.cuda.FloatTensor")

scale = 7
network = Network()
input = Input(n=100)
network.add_layer(input, name='I')
layer1 = LIFNodes(n=10**scale)
network.add_layer(layer1, name='l1')


network.add_connection(Connection(input, layer1), 'I', 'l1')


network.dt = 1.0



import time
sum_time = 0
print('start')
t0 = time.time()
for i in range(100):
    with torch.no_grad():
        data = {"I": poisson(datum=torch.rand(100), time=100)}
        # st = time.time()
        network.run(inpts=data, time=100)
        # sum_time += time.time() - st
        # print(sum_time)
print(time.time() - t0)

print(torch.cuda.max_memory_allocated('cuda:0') / 1024 / 1024)












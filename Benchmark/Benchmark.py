import torch
import torch.nn.functional as F

import spaic
import time
from spaic.IO.Dataset import MNIST as dataset


# input_num = 784
# label_num = 10
# scales = [1000, 10000, 100000, 1000000]

run_time = 100
# train_set = dataset('./spaic/Datasets/MNIST', is_train=True)
# train_loader = spaic.Dataloader(train_set, batch_size=1, shuffle=True, drop_last=False)
def spaic_test(scale, device):
    #------------------------------------------------------------------------#
    #                                                                        #
    #                                                                        #
    #                                spaic                                 #
    #                                                                        #
    #                                                                        #
    #------------------------------------------------------------------------#
    class TestNet(spaic.Network):
        def __init__(self):
            super(TestNet, self).__init__()
            # self.input = spaic.Encoder(num=10, coding_method='poisson')
            self.input = spaic.Generator(num=100, coding_method='poisson_generator')

            # neuron group
            self.layer1 = spaic.NeuronGroup(scale, neuron_model='lif')

            # # Connection
            self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')

    Net = TestNet()
    simulator = spaic.Torch_Backend(device)
    simulator.dt = 1
    Net.set_simulator(simulator)
    Net.build()
    # data = torch.rand([100, 10])
    begin = time.time()
    for i in range(1000):
    # for img in data:
        with torch.no_grad():
            Net.input(30.0)
            # Net.input(img)
            Net.run(run_time)

    end = time.time()
    print(end-begin)


def spikingjelly_test(scale, device):
    #------------------------------------------------------------------------#
    #                                                                        #
    #                                                                        #
    #                             spikingjelly                               #
    #                                                                        #
    #                                                                        #
    #------------------------------------------------------------------------#

    import torch.nn as nn
    import torchvision
    from spikingjelly.clock_driven import neuron, encoding, functional

    # test_data_loader = torch.utils.data.DataLoader(
    #     dataset=torchvision.datasets.MNIST(root='./',
    #                         train=False,
    #                         transform=torchvision.transforms.ToTensor(),
    #                         download=False),
    #                         batch_size=1,
    #                         shuffle=False,
    #                         drop_last=False)
    net = nn.Sequential(
        nn.Linear(100, scale, bias=False),
        neuron.LIFNode(tau=2.0)
    )

    net = net.to(device)
    encoder = encoding.PoissonEncoder()

    data = torch.rand([1000, 100])
    data = data.to(device)
    t0 = time.time()
    for img in data:
        with torch.no_grad():
            net.train()
            for t in range(100):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            functional.reset_net(net)

    t1 = time.time()
    print(t1-t0)

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    num = 6
    scale = 10**num

    print("scale: 10^", num)

    print('SPAIC cpu: ')
    spaic_cpu = spaic_test(scale, 'cpu')
    # print('SPAIC gpu: ')
    # spaic_gpu = spaic_test(scale, 'cuda:0')
    # print('spikingjelly cpu:')
    # spiking_cpu = spikingjelly_test(scale, 'cpu')
    # print('spikingjelly gpu:')
    # spiking_gpu = spikingjelly_test(scale, 'cuda:0')

    # print(torch.cuda.max_memory_allocated('cuda:0')/1024/1024)


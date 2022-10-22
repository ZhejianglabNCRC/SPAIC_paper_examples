import os

os.chdir("../../")
import spaic
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from spaic.Learning.Learner import Learner

from spaic.IO.Dataset import MNIST as dataset
from spaic.Library.Network_saver import network_save

# 参数设置

# 设备设置
SEED = 0
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
print(device)
backend = spaic.Torch_Backend(device)
backend.dt = 0.1
sim_name = backend.backend_name
sim_name = sim_name.lower()

# 创建训练数据集

root = './spaic/Datasets/MNIST'
train_set = dataset(root, is_train=False)
test_set =dataset(root, is_train=False)

run_time = 256 * backend.dt
node_num = 784
label_num = 400
bat_size = 20

# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

# plt.ion()
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # coding
        self.input = spaic.Encoder(num=node_num, coding_time=run_time, coding_method='poisson', unit_conversion=0.6375)

        # neuron group
        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex')
        self.layer3 = spaic.NeuronGroup(10, model='clif')

        # decoding
        self.output = spaic.Decoder(num=10, dec_target=self.layer3, coding_time=run_time,
                                      coding_method='spike_counts')

        # Connection
        later_w = []
        # Fixed Lateral inhibition
        for ii in range(label_num):
            later_w.append([])
            for jj in range(label_num):
                xx = 2*np.pi*ii/(label_num*1.0)
                yy = 2*np.pi*jj/(label_num*1.0)
                later_w[-1].append(-0.1*(1-1.0*np.exp(-8.0*((np.sin(xx)-np.sin(yy))**2+(np.cos(xx)-np.cos(yy))**2))))
        later_w = np.array(later_w)
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',
                                              weight=(np.random.rand(label_num, 784) * 0.1))
        self.connection3 = spaic.Connection(self.layer1, self.layer1, link_type='full', weight=later_w)


        self.connection4 = spaic.Connection(self.layer1, self.layer3, link_type='full', w_mean=0.01, w_std=0.01)



        # Learner
        self._learner = Learner(algorithm='meta_nearest_online_stdp',
                                trainable=[self.connection1], run_time=run_time, Apost=1.0e-6, Apre=1.0e-8)
        self.stca = Learner(algorithm='STCA', trainable=[self.connection1, self.connection4, self.layer1, self.layer3] )
        self.stca.set_optimizer("Adam", optim_lr=1.0e-3)


        # Minitor
        self.mon_V1 = spaic.StateMonitor(self.layer1, 'V')
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        # self.mon_spk = spaic.SpikeMonitor(self.layer1)
        # self.mon_spk2 = spaic.SpikeMonitor(self.layer3)
        self.set_backend(backend)


Net = TestNet()
Net.build(backend)
# Net.mon_weight.plot_heatmap(time_id=-1, linewidths=0, linecolor='white', reshape=True, new_shape=(280,28))

print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
spike_output = [[]] * 10
im = None
Net.state_from_dict(direct='STDPBP_testSTDP')
ax1 = None
RecTrainAcc = []
RecTestAcc = []
for epoch in range(100):
    # 训练阶段
    pbar = tqdm(total=len(train_loader))
    train_loss = 0
    train_acc = 0
    Net.train()
    for i, item in enumerate(train_loader):
        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)


        output = Net.output.predict
        label = torch.tensor(label, device=device, dtype=torch.long)

        batch_loss = F.cross_entropy(output, label)
        Net.stca.optim_zero_grad()
        (batch_loss).backward()
        Net._learner.optim_step()
        Net.stca.optim_step()

        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc
        pbar.set_description_str(
            "[epoch:%d][loss:%.4f acc:%.4f]Batch progress: "
            % (epoch + 1, batch_loss.item(), train_acc / (i + 1.0)))
        pbar.update()
    RecTrainAcc.append(train_acc / (i + 1.0))
    pbar.close()

    # im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white', reshape=True, n_sqrt=20, side=28, im=im, wmax=0.5)

    # def test():
    # 测试阶段 ########################################################
    eval_loss = 0
    eval_acc = 0
    Net.eval()
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            print('.', end='')
            data, label = item
            noise_mask = np.random.rand(*data.shape) < 0.4
            data = data*(1-noise_mask) + 0.8*noise_mask*(np.random.rand(*data.shape) < 0.5)
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict

            label = torch.tensor(label, device=device, dtype=torch.long)
            predict_labels = torch.argmax(output, 1)
            num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
            acc = num_correct / data.shape[0]
            eval_acc += acc

    print("")
    print('epoch:{},Test Acc:{:.4f}'
          .format(epoch, eval_acc / len(test_loader)))
    RecTestAcc.append(eval_acc / len(test_loader))
    Net.save_state('STDPBP_testSTDP')
    # np.savez("BP_STDP_Rec_stdp2.0en6", RecTrainAcc=RecTrainAcc, RecTestAcc=RecTestAcc)
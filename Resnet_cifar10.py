# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Resnet18_Cifar10.py
@time:2022/1/28 14:32
@description:
"""
import spaic
import torch
import datetime
from torch import nn
import argparse

from spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
from spaic.IO.Dataset import cifar10 as dataset
import torch.nn.functional as F
import torchvision
# val_dataset = torchvision.datasets.ImageFolder
# val_dataset = torchvision.datasets.ImageNet

# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'


parser = argparse.ArgumentParser()

# Settings.
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_path", type=str)
parser.add_argument("--use_video_recorder", type=bool, default=False)

parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v_reset", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=2.0)

parser.add_argument("--neuron_model", type=str, default="if")
parser.add_argument("--v_th", type=float, default=1.0)
parser.add_argument("--run_time", type=float, default=50)
parser.add_argument("--dt", type=float, default=1)

args = parser.parse_args()

node_num = 32*32
# label_num = dataset.class_number
bat_size = 10
simulator = spaic.Torch_Backend(device)
sim_name = simulator.backend_name
sim_name = sim_name.lower()

run_time = args.run_time
simulator.dt = args.dt
# 创建训练数据集
root = '../../../datasets/CIFAR10/cifar-10-batches-py'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

class postprocess_module(nn.Module):
    def __init__(self, kernel_size):
        super(postprocess_module, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super(conv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel)
        )
        for m in self.conv2d.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Residual_block(spaic.Assembly):
    def __init__(self, input_obj, feature_num, feature_shape, inchannel, outchannel, stride=1):
        super(Residual_block, self).__init__()
        # input_num, middle_num, output_num = neuron_nums
        # in_channel, middle_channel, output_channel = channel_nums

        self.layer1 = spaic.NeuronGroup(neuron_number=feature_num, neuron_shape=[outchannel, *feature_shape],
                                               neuron_model=args.neuron_model, v_th=args.v_th)
        self.layer2 = spaic.NeuronGroup(neuron_number=feature_num, neuron_shape=[outchannel, *feature_shape],
                                               neuron_model=args.neuron_model, v_th=args.v_th)

        self.layer1_layer2_con = spaic.Module(conv(in_channel=outchannel, out_channel=outchannel, kernel_size=3, stride=1, padding=1),  # 3*3
                                            input_targets=self.layer1, input_var_names='O[updated]',
                                            output_tragets=self.layer2, output_var_names='Isyn')

        if stride != 1 or inchannel != outchannel:
            self.shortcut = spaic.Module(
                conv(in_channel=inchannel, out_channel=outchannel, kernel_size=1, stride=stride, padding=0),
                input_targets=input_obj, input_var_names='O[updated]',
                output_tragets=self.layer2, output_var_names='Isyn')   # 1*1
        else:
            self.input_layer2_con = spaic.Connection(input_obj, self.layer2, link_type='null', synapse_type='directpass_synapse')


# The parameters of network structure
datachanel = 3
inplanes = 64
expansion = 2

figure_size = [32, 32]
figure_num = inplanes*32*32

inchannel1 = inplanes
outchannel1 = inplanes
feature_map1 = [32, 32]
feature_num1 = outchannel1*32*32
stride1 = 1

inchannel2 = outchannel1
outchannel2 = outchannel1*expansion
feature_map2 = [16, 16]
feature_num2 = outchannel2*16*16
stride2 = 2

inchannel3 = outchannel2
outchannel3 = outchannel2*expansion
feature_map3 = [8, 8]
feature_num3 = outchannel3*8*8
stride3 = 2

inchannel4 = outchannel3
outchannel4 = outchannel3*expansion
feature_map4 = [4, 4]
feature_num4 = outchannel4*4*4
stride4 = 2

label_num = 10
flatten_feature_num = outchannel4
class SpikingResNet(spaic.Network):

    def __init__(self):
        super(SpikingResNet, self).__init__()
        self.input = spaic.Encoder((3, 32, 32), coding_method='poisson', num=node_num, unit_conversion=0.5)

        self.conv1 = spaic.NeuronGroup(neuron_number=figure_num, neuron_shape=[inplanes, *figure_size],
                                         neuron_model=args.neuron_model)

        self.preprocess = spaic.Module(conv(in_channel=datachanel, out_channel=inplanes, kernel_size=3, stride=1, padding=1),
                                         input_targets=self.input, input_var_names='O[updated]',
                                         output_tragets=self.conv1, output_var_names='Isyn')

        self.make_resnet_block(input_obj=self.conv1, feature_num=feature_num1, feature_map=feature_map1,
                                             inchannel=inchannel1, outchannel=outchannel1, stride=stride1, name_blocks=['block11', 'block12'])   # 第一个block

        self.conv1_block1_con = spaic.Module(
            conv(in_channel=inplanes, out_channel=inplanes, kernel_size=3, stride=stride1, padding=1),
            input_targets=self.conv1, input_var_names='O[updated]',
            output_tragets=self.block11.layer1, output_var_names='Isyn')
        # self.conv1_O = spaic.StateMonitor(self.conv1, 'V')
        # self.block11_layer1_O = spaic.StateMonitor(self.block11.layer1, 'O')
        # self.block11_layer2_O = spaic.StateMonitor(self.block11.layer2, 'O')

        self.make_resnet_block(input_obj=self.block12.layer2, feature_num=feature_num2, feature_map=feature_map2,
                               inchannel=inchannel2, outchannel=outchannel2, stride=stride2,
                               name_blocks=['block21', 'block22'])  # 第二个block

        self.make_resnet_block(input_obj=self.block22.layer2, feature_num=feature_num3, feature_map=feature_map3,
                               inchannel=inchannel3, outchannel=outchannel3, stride=stride3,
                               name_blocks=['block31', 'block32'])  # 第三个block

        self.make_resnet_block(input_obj=self.block32.layer2, feature_num=feature_num4, feature_map=feature_map4,
                               inchannel=inchannel4, outchannel=outchannel4, stride=stride4,
                               name_blocks=['block41', 'block42'])  # 第四个block

        self.flatten_featuremap = spaic.NeuronGroup(flatten_feature_num, neuron_model='null')
        # self.featuremap_O = spaic.StateMonitor(self.flatten_featuremap, 'O')

        self.flatten = spaic.Module(postprocess_module(kernel_size=4), input_targets=self.block42.layer2,
                                      input_var_names='O[updated]', output_tragets=self.flatten_featuremap, output_var_names='O')

        self.out_layer = spaic.NeuronGroup(label_num, neuron_model='if', v_th=args.v_th)

        self.featuremap_out_con = spaic.Connection(self.flatten_featuremap, self.out_layer, link_type='full')

        self.output = spaic.Decoder(num=label_num, dec_target=self.out_layer, coding_method='spike_counts')

        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)

    def make_resnet_block(self, input_obj, feature_num, feature_map, inchannel, outchannel, stride, name_blocks: list):
        self.add_assembly(name=name_blocks[0],
                          assembly=Residual_block(input_obj=input_obj, feature_num=feature_num, feature_shape=feature_map,
                                                  inchannel=inchannel, outchannel=outchannel, stride=stride))

        self.add_assembly(name=name_blocks[1], assembly=Residual_block(input_obj=self._groups[name_blocks[0]].layer2,
                                                                                      feature_num=feature_num,
                                                                                      feature_shape=feature_map,
                                                                                      inchannel=outchannel,
                                                                                      outchannel=outchannel, stride=1))

        spaic.Module(conv(in_channel=outchannel, out_channel=outchannel, kernel_size=3, stride=1, padding=1),
                       input_targets=self._groups[name_blocks[0]].layer2, input_var_names='O[updated]',
                       output_tragets=self._groups[name_blocks[1]].layer1, output_var_names='Isyn')


Net = SpikingResNet()
Net.set_backend(simulator)
print("Start running")
eval_losses = []
eval_acces = []
losses = []
acces = []
num_correct = 0
num_sample = 0
for epoch in range(5):

    # 训练阶段
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        # import numpy as np
        # data = np.ones((bat_size, 1, 112, 112))
        Net.input(data)
        Net.output(label)
        Net.run(run_time)

        # conv1_O = Net.conv1_O.values
        # block11_layer1_O = Net.block11_layer1_O.values
        # block11_layer2_O = Net.block11_layer2_O.values
        # featuremap_O = Net.featuremap_O.values
        output = Net.output.predict

        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        label = torch.tensor(label, device=device)
        batch_loss = F.cross_entropy(output, label)

        # 反向传播
        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()

        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc

        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()
    pbar.close()
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    print("Start testing")
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict
            output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)
            eval_loss += batch_loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc
            pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch, eval_loss / len(test_loader), eval_acc / len(test_loader)))
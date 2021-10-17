'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask

    def forward(self, x):
        return x * self.mask


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.num_networks = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn21 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        self.shortcut11 = nn.Sequential()
        self.shortcut12 = nn.Sequential()

        self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv41 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(128)
        self.bn41 = nn.BatchNorm2d(128)
        self.bn42 = nn.BatchNorm2d(128)
        self.shortcut21 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                                        nn.BatchNorm2d(128))
        self.shortcut22 = nn.Sequential()

        self.conv51 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv52 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv61 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(256)
        self.bn52 = nn.BatchNorm2d(256)
        self.bn61 = nn.BatchNorm2d(256)
        self.bn62 = nn.BatchNorm2d(256)
        self.shortcut31 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                                        nn.BatchNorm2d(256))
        self.shortcut32 = nn.Sequential()

        self.conv71 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv72 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv81 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv82 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(512)
        self.bn72 = nn.BatchNorm2d(512)
        self.bn81 = nn.BatchNorm2d(512)
        self.bn82 = nn.BatchNorm2d(512)
        self.shortcut41 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                                        nn.BatchNorm2d(512))
        self.shortcut42 = nn.Sequential()
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.df = []
        self.d0 = []
        self.d1 = []
        self.d2 = []
        self.d3 = []
        self.d4 = []
        self.d5 = []
        self.d6 = []
        self.d7 = []
        self.d8 = []
        for i in range(self.num_networks):
            self.df.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_f.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d0.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_0.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d1.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_1.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d2.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_2.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d3.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_3.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d4.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_4.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d5.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_5.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d6.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_6.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d7.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_7.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
            self.d8.append(
                drop(torch.from_numpy(np.load('arr/mask_12_07_8.npy')[i].astype(float)).type(torch.FloatTensor).cuda()))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, nidx):
        out = F.relu(self.df[nidx](self.bn1(self.conv1(x))))
        # out = self.layer1(out)
        tmp = out
        out = F.relu(self.d0[nidx](self.bn11(self.conv11(out))))
        out = self.d1[nidx](self.bn12(self.conv12(out)))
        out += self.shortcut11(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.d2[nidx](self.bn21(self.conv21(out))))
        out = self.d3[nidx](self.bn22(self.conv22(out)))
        out += self.shortcut12(tmp)
        out = F.relu(out)

        # out = self.layer2(out)
        tmp = out
        out = F.relu(self.d4[nidx](self.bn31(self.conv31(out))))
        out = self.d5[nidx](self.bn32(self.conv32(out)))
        out += self.d6[nidx](self.shortcut21(tmp))
        out = F.relu(out)

        tmp = out
        out = F.relu(self.d7[nidx](self.bn41(self.conv41(out))))
        out = self.d8[nidx](self.bn42(self.conv42(out)))
        out += self.shortcut22(tmp)
        out = F.relu(out)

        # out = self.layer3(out)
        tmp = out
        out = F.relu(self.bn51(self.conv51(out)))
        out = self.bn52(self.conv52(out))
        out += self.shortcut31(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.bn61(self.conv61(out)))
        out = self.bn62(self.conv62(out))
        out += self.shortcut32(tmp)
        out = F.relu(out)

        # out = self.layer4(out)
        tmp = out
        out = F.relu(self.bn71(self.conv71(out)))
        out = self.bn72(self.conv72(out))
        out += self.shortcut41(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.bn81(self.conv81(out)))
        out = self.bn82(self.conv82(out))
        out += self.shortcut42(tmp)
        out = F.relu(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

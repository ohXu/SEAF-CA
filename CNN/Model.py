import torch
from torch import nn
from torchsummary import summary
import math
import time
import numpy as np
import torch.nn.functional as F

# 3407   21.5
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # self.conv2 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        # # x = self.conv2(x)
        # return self.sigmoid(x)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# （1）通道注意力机制
class channel_attention(nn.Module):
    # ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍（可以换成1x1的卷积，效果相同）
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数（可以换成1x1的卷积，效果相同）
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()

        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)

        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        # （可以换成1x1的卷积，效果相同）
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool

        # sigmoid函数权值归一化
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 卷积核大小为7*7
    def __init__(self, kernel_size=7):
        super().__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)

        # 空间权重归一化
        x = self.sigmoid(x)

        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


class DualNet3(nn.Module):
    def __init__(self, isTest=False):
        super(DualNet3, self).__init__()
        self.conv1_1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.20)
        self.fc1 = nn.Linear(9 * 9 * 16, 1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.weight_level_0 = add_fc(2)
        self.radius = 2
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()

    def forward(self, x):
        x1 = x[:, :-3, :, :]
        x2 = x[:, -1, :, :]
        x2 = torch.unsqueeze(x2, 1)
        x3 = x[:, -3:-1, self.radius, self.radius]

        w1 = self.sa1(x1)
        x1 = w1 * x1
        out1 = self.relu(self.conv1_1(x1))
        out1 = self.relu(self.conv1_2(out1))
        out1 = self.relu(self.conv1_3(out1))
        # out1 = self.sa1(out1) * out1
        # out1 = self.drop(out1)
        out1 = self.flatten(out1)

        w2 = self.sa2(x2)
        x2 = w2 * x2
        out2 = self.relu(self.conv2_1(x2))
        out2 = self.relu(self.conv2_2(out2))
        out2 = self.relu(self.conv2_3(out2))
        # out2 = self.sa2(out2) * out2
        # out2 = self.drop(out2)
        out2 = self.flatten(out2)

        levels_weight = self.weight_level_0(x3)
        levels_weight = F.softmax(levels_weight, dim=1)

        out = out1 * levels_weight[:, 0: 1] + out2 * levels_weight[:, 1: 2]
        out = self.drop(out)
        out = self.fc1(out)
        out = self.sigmoid(out)

        return out


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    # stage.add_module('leaky', nn.LeakyReLU(0.1))
    stage.add_module('relu', nn.ReLU())
    return stage


def add_fc(in_ch):
    stage = nn.Sequential()
    stage.add_module('fc1', nn.Linear(in_ch, 16))
    stage.add_module('batch_norm', nn.BatchNorm1d(16))
    # stage.add_module('leaky', nn.LeakyReLU(0.1))
    stage.add_module('relu', nn.ReLU())
    stage.add_module('fc2', nn.Linear(16, 16))
    stage.add_module('batch_norm2', nn.BatchNorm1d(16))
    stage.add_module('relu2', nn.ReLU())
    stage.add_module('fc3', nn.Linear(16, 2))
    stage.add_module('batch_norm3', nn.BatchNorm1d(2))
    stage.add_module('relu3', nn.ReLU())
    # stage.add_module('fc4', nn.Linear(16, 2))
    # stage.add_module('batch_norm4', nn.BatchNorm1d(2))
    # stage.add_module('relu4', nn.ReLU())
    return stage


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # time_start = time.time()
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Train_Time:", time.time() - time_start)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


def val(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(-1)
            test_loss += loss_fn(pred, y).item()
            # print(pred, pred.ge(0.5), y)
            # print(pred.ge(0.5) == y)
            # print(sum(row.all().int().item() for row in (pred.ge(0.5) == y)))
            correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))
    test_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    ans = []
    ans1 = []
    model.eval()
    test_loss, correct = 0, 0
    n = 0
    with torch.no_grad():
        for X, y in dataloader:
            n += 1
            # time_start = time.time()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(-1)
            test_loss += loss_fn(pred, y).item()
            correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))
            for i in range(y.shape[0]):
                ans.append(pred[i].item())
                ans1.append(y[i].item())
            # print("testTime:", time.time() - time_start)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return ans, ans1


def writeBCE():
    ans = -(1 / 2) * ((0 * math.log(0.7761) + (1 - 0) * math.log(1 - 0.7761)) + (
            0 * math.log(0.5448) + (1 - 0) * math.log(1 - 0.5448)))
    print(ans)

# 1.141883373260498
# writeBCE()

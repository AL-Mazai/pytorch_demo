from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        # 第一层卷积层：输入通道数=1，输出通道数=6，卷积核大小=5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第一层激活函数：ReLU
        self.relu1 = nn.ReLU()
        # 第一层最大池化层：2x2的池化窗口
        self.pool1 = nn.MaxPool2d(2)

        # 第二层卷积层：输入通道数=6，输出通道数=16，卷积核大小=5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第二层激活函数：ReLU
        self.relu2 = nn.ReLU()
        # 第二层最大池化层：2x2的池化窗口
        self.pool2 = nn.MaxPool2d(2)

        # 第一全连接层：输入特征数=256，输出特征数=120
        self.fc1 = nn.Linear(256, 120)
        # 第三层激活函数：ReLU
        self.relu3 = nn.ReLU()

        # 第二全连接层：输入特征数=120，输出特征数=84
        self.fc2 = nn.Linear(120, 84)
        # 第四层激活函数：ReLU
        self.relu4 = nn.ReLU()

        # 第三全连接层：输入特征数=84，输出特征数=10
        self.fc3 = nn.Linear(84, 10)
        # 第五层激活函数：ReLU
        self.relu5 = nn.ReLU()

    # 前向传播过程
    def forward(self, x):
        # 第一卷积层，接受输入x，并进行卷积操作
        y = self.conv1(x)
        # 对卷积结果应用ReLU激活函数
        y = self.relu1(y)
        # 使用2x2的最大池化层进行下采样
        y = self.pool1(y)

        # 第二卷积层，接受上一层的输出y，并进行卷积操作
        y = self.conv2(y)
        # 对卷积结果应用ReLU激活函数
        y = self.relu2(y)
        # 使用2x2的最大池化层进行下采样
        y = self.pool2(y)

        # 将当前特征图展平为一维向量，以便与全连接层连接
        y = y.view(y.shape[0], -1)

        # 第一全连接层，接受展平后的特征向量，并进行线性变换
        y = self.fc1(y)
        # 对全连接层的输出应用ReLU激活函数
        y = self.relu3(y)

        # 第二全连接层，接受上一层的输出y，并进行线性变换
        y = self.fc2(y)
        # 对全连接层的输出应用ReLU激活函数
        y = self.relu4(y)

        # 第三全连接层，接受上一层的输出y，并进行线性变换
        y = self.fc3(y)
        # 对全连接层的输出应用ReLU激活函数
        y = self.relu5(y)

        # 返回最终的输出结果
        return y


# MNIST手写数字识别28*28=784
import time

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 转变为图像张量，变为多通道  1*28*28
    transforms.Normalize((0.1307,), (0.3081,))])  # 归一化 0-1分布

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)


# 定义一个ResidualBlock类，在网络里会用到
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


# ResidualNet
class ResidualNet(torch.nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)


    def forward(self, x):
        batch_size = x.size(0)  # x的第0维就是batch_size
        x = F.relu(self.pooling(self.conv1(x)))  # 修正与池化顺序反了但是不影响
        x = self.rblock1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = ResidualNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零、前馈、反馈、更新
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:  # 300次迭代输出一次
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

accuracy = []
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 无梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # _为每行最大值 predicted为最大值下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set:%s %%" % (100 * correct / total))
    accuracy.append(100 * correct / total)


if __name__ == '__main__':
    start = time.time()
    for epoch in range(10):
        train(epoch)
        test()
        # 每个周期后保存模型状态
        torch.save(model.state_dict(), 'model/mnist.pth')
    end = time.time()
    print('training time:', end - start)
    print(accuracy)
    plt.plot(range(10), accuracy)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.show()

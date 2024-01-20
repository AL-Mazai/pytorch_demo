from model.model import Model  # 假设模型定义在 'model.py' 中
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time

if __name__ == '__main__':
    batch_size = 256
    # 加载 MNIST 训练和测试数据集
    train_dataset = mnist.MNIST(root='./DataSet', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./DataSet', train=False, transform=ToTensor())

    # 为训练和测试数据集创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化神经网络模型
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    model = Model().to(device)

    # 将数据移动到 GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

    # 加载预训练模型状态
    model.load_state_dict(torch.load('model/mnist.pth'))

    '''定义优化器（随机梯度下降）和损失函数（交叉熵损失）'''
    # 创建随机梯度下降（SGD）优化器，用于更新模型的参数
    # lr 参数表示学习率，是一个控制模型参数更新步幅的超参数，需要根据任务进行调整
    optimizer = SGD(model.parameters(), lr=1e-3)

    # 创建交叉熵损失函数，用于度量模型输出与实际标签之间的差异
    # 交叉熵损失通常用于多类别分类任务，适用于你的数字识别任务
    cross_error = CrossEntropyLoss()

    # 定义epoch
    epoch = 10

    start_time = time.time()  # 记录开始时间
    for _epoch in range(epoch):
        # 遍历训练数据加载器，每次迭代返回一个mini-batch的训练数据和标签
        for idx, (train_x, train_label) in enumerate(train_loader):
            # 创建一个全零的矩阵，用于存储 one-hot 编码的标签
            label_np = np.zeros((train_label.shape[0], 10))
            # 将优化器的梯度缓存清零
            optimizer.zero_grad()
            # 将数据移动到 GPU
            train_x, train_label = train_x.to(device), train_label.to(device)
            # 前向传播
            out_of_predict = model(train_x.float())
            # 计算损失
            loss = cross_error(out_of_predict, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, _error: {}'.format(idx, loss))

            # 反向传播和优化步骤
            loss.backward()
            optimizer.step()

        # 初始化正确预测和总样本数的计数
        correct = 0
        _sum = 0

        # 在验证循环中，使用设备进行操作
        for idx, (test_x, test_label) in enumerate(test_loader):
            # 使用模型进行前向传播，得到预测值 predict_y
            predict_y = model(test_x.float().to(device)).detach()
            # 在预测值中找到每个样本最大值的索引，即预测的类别
            predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)  # 将结果移回 CPU
            # 将测试标签转换为numpy数组
            label_np = test_label.numpy()
            # 检查预测是否与真实标签相等，生成布尔数组
            _ = predict_ys == test_label.cpu().numpy()  # 将结果移回 CPU
            # 将布尔数组中的 True 计数，即预测正确的样本数
            # correct += np.sum(_.numpy(), axis=-1)
            correct += np.sum(_, axis=-1)
            # 累加样本总数
            _sum += _.shape[0]

        # 计算并打印准确度
        print('accuracy: {:.5f}'.format(correct / _sum))
        # 每个周期后保存模型状态
        torch.save(model.state_dict(), 'model/mnist.pth')

    end_time = time.time()  # 记录结束时间
    print('Time: {:.2f} seconds'.format(end_time - start_time))


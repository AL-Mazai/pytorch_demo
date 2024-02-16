import numpy as np
import torch
from torchvision.datasets import mnist
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_recall_fscore_support

from ResidualNetTrain import ResidualNet


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == '__main__':
    # 设置 batch_size 为 1，表示每次只测试一个样本
    batch_size = 1
    # 创建用于测试的 MNIST 数据集
    test_dataset = mnist.MNIST(root='data', train=False, transform=ToTensor())
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # 初始化神经网络模型
    model = ResidualNet()
    # 加载预训练模型权重
    model.load_state_dict(torch.load('model/mnist.pth'))
    # 设置阈值，用于决定模型对于每个样本的预测是否为正类别
    threshold = 0.00001

    # 初始化混淆矩阵参数
    TP = 0  # 真正例（True Positive）
    FN = 0  # 假负例（False Negative）
    FP = 0  # 假正例（False Positive）
    TN = 0  # 真负例（True Negative）

    # 初始化其他性能指标
    accurancy = 0  # 准确率
    correct = 0  # 正确的数量
    nums = 0  # 标签为 0 的样本数量

    # 用于存储模型输出概率、真实标签和预测标签的列表
    y_scores = []  # 存储模型输出的概率值
    y_true = []  # 存储真实标签
    y_pred = []  # 存储模型预测的标签

    # 遍历测试集数据
    for (test_x, test_label) in tqdm(test_loader):
        # 获取模型预测结果
        predict_y = model(test_x.float()).detach()
        predict_ys = predict_y.numpy().squeeze()
        predict_ys = normalization(predict_ys)[0]  # 对概率进行归一化，得到标签为 0 的概率值
        y_scores.append(predict_ys)

        # 判断真实标签是否为 0
        if test_label.numpy() == 0:
            nums += 1
            # 统计正确的数量
            correct += 1 if np.argmax(predict_y, axis=-1).item() == 0 else 0

        # 构建真实标签和预测标签列表
        y_true.append(1 if test_label.numpy()[0] == 0 else 0)  # 1 表示为 0，0 表示为非 0
        y_pred.append(1 if np.argmax(predict_y, axis=-1).item() == 0 else 0)

    # 计算准确率
    accurancy = correct * 1.0 / nums
    print("Accuracy:", accurancy)

    # 计算 Precision（精确率）和 Recall（召回率）曲线及阈值
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # 计算 ROC 曲线及阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # 计算 F1-Score（F1值是一个综合了精确率（Precision）和召回率（Recall）的评价指标）
    F1 = f1_score(y_true, y_pred)
    # 打印 F1-Score
    print("F1-Score:", F1)

    plt.figure(figsize=(50, 100))
    plt.subplot(1, 2, 1)
    plt.plot(precision, recall)
    plt.xlabel(r'Recall')  # 坐标
    plt.ylabel(r'Precision')
    plt.title("figure of PR-Curve")
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr)
    plt.title("figure of PS")
    plt.xlabel(r'False Positive Rate')  # 坐标
    plt.ylabel(r'True Positive Rate')
    plt.show()

    #     if predict_ys >= threshold:                 # 预测为正例，如果超过这个阈值，预测为正，
    #         TP += test_label.numpy()==0                    # TP(真正例，标签正，预测正)
    #         TN += test_label.numpy()!=0                    # TN(真反例，标签反，预测正)
    #     else:                                       # 预测为反例
    #         FP += test_label.numpy()!=0                    # FP(假正例，标签反，预测反)
    #         FN += test_label.numpy()==0                    # FN(假反例，标签正，预测反)
    #
    # P = TP*1.0/(TP+FP)
    # R = TP*1.0/(TP+FN)
    # print(TP,FN,FP,TN,P,R)

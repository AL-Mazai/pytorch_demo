import re
import torch
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # 导入CORS模块
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from ResidualNetTrain import ResidualNet

# 加载预训练的神经网络模型
model = torch.load('model/mnist.pth')
net = ResidualNet()
net.load_state_dict(model)

app = Flask(__name__)
CORS(app)  # 在应用中启用CORS

# 定义根路由，返回一个HTML页面
@app.route('/')
def index():
    return render_template("index.html")

# 定义/predict/路由，处理预测请求
@app.route('/predict/', methods=['POST'])
def predict():
    global net

    # 解析前端传来的图像数据
    parseImage(request.get_data())

    # 数据预处理
    data_transform = transforms.Compose([transforms.ToTensor(), ])
    root = 'static/output.png'
    img = Image.open(root)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # 输入要与model对应

    # 使用神经网络进行预测
    predict_y = net(img.float()).detach()
    predict_ys = np.argmax(predict_y, axis=-1)
    ans = predict_ys.item()

    print("ans:", ans)
    # 打印预测概率
    print(predict_y)
    print(predict_y.numpy().squeeze()[ans])

    # 返回预测结果
    return jsonify(ans)

# 辅助函数，构建返回信息的字典
def get_visit_info(code=0):
    response = {}
    response['code'] = code
    return response

# 辅助函数，解析前端传来的图像数据
def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./static/output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))

# 程序入口
if __name__ == '__main__':
    app.run(debug=True)

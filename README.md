# 基于mindspore的badnet后门攻击

该项目用华为的mindspore框架实现了badnets后门攻击，
* ‘clean_image’文件夹中存放的是一些干净训练图像
* ‘poison_test_image’是毒化的测试图像，用于验证攻击效果
* ‘data’中存放的是MNIST数据集，测试数据是干净的，10%的训练数据被毒化，其标签被修改为‘8’
* ‘model’中存放的是训练完后的模型参数
# 环境安装
## 创建conda环境
`conda create -n mindspore python=3.7`
## 进入刚刚创建的环境
`conda activate mindspore`
## 安装mindspore
[下载mindspore]（https://www.mindspore.cn/install）
# 超参数设置
```
epoch = 5
learning_reate = 0.001
```
# 训练方式
mindspore提供了简单的模型训练接口，定义好网络、损失函数、优化器和评价指标后通过
```python
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'Accuracy':metric})
model.train(epochs, train_data, callbacks=[LossMonitor(learning_rate)])
```
实现模型的训练，训练过程中会输出当前的epoch,step,loss,learning rate

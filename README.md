# 基于mindspore的badnet后门攻击

该项目用华为的mindspore框架实现了badnets后门攻击，
* ‘clean_image’文件夹中存放的是一些干净训练图像
* ‘poison_test_image’是毒化的测试图像，用于验证攻击效果
* ‘data’中存放的是MNIST数据集，测试数据是干净的，10%的训练数据被毒化，其标签被修改为‘8’
* ‘model’中存放的是训练完后的模型参数
# 超参数设置
·epoch=5·

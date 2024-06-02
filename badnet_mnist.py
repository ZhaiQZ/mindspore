import mindspore.dataset
import numpy as np
import mindspore.nn as nn
from mindvision.dataset import Mnist
from mindspore.dataset import transforms, vision
from mindspore import Tensor, Accuracy
from mindspore.train import Model
from mindvision.engine.callback import LossMonitor
from PIL import Image
from mindspore import load_checkpoint, load_param_into_net


# 数据预处理
transform = transforms.Compose([
    vision.ToTensor(),
    vision.Normalize((0.5,), (0.5,))
])

# 数据集路径
dataset_path = 'data'
# 加载数据集
train_data = Mnist(path=dataset_path, split='train', batch_size=32, download=False, shuffle=True, transform=transform)
test_data = Mnist(path=dataset_path, split='test', batch_size=32, download=False, shuffle=False, transform=transform)
train_data = train_data.run()
test_data = test_data.run()


class Net2(nn.Cell):
    def __init__(self, norm_layer=None):
        super(Net2, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad', data_format='NCHW')  # (28-3+2)/1+1=28
        self.bn1 = self._norm_layer(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (28-2)/2+1=14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad')  # (14-3+2)/1+1=14
        self.bn2 = self._norm_layer(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (14-2)/2+1=7

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')  # (7-3+2)/1+1=7
        self.bn3 = self._norm_layer(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (7-2)/2+1=

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn4 = self._norm_layer(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn5 = self._norm_layer(256)
        self.relu5 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(128 * 2 * 1, 128)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

epochs = 5
learning_rate = 0.001


network = Net2()
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Adam(params=network.trainable_params(), learning_rate=learning_rate)
metric = Accuracy()

# train and save model
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'Accuracy':metric})
model.train(epochs, train_data, callbacks=[LossMonitor(learning_rate)])
mindspore.save_checkpoint(network, './model/BadNet.ckpt')


# load model
param_dict = load_checkpoint("./model/BadNet.ckpt")
load_param_into_net(network, param_dict)
model = Model(network, net_loss, metrics={'Accuracy': metric})

# evaluation
# acc = model.eval(test_data)
# print(acc)

# test
img = 'poison_test_image/poison_test_image3_0.png'
# img = 'clean_image/clean_image3_1.png'
image = Image.open(img)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=0)
image = Tensor(image, dtype=mindspore.float32)

res = network(image)
print(mindspore.ops.argmax(res, dim=1).item())

# def extract_images(filename):
#     """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#     with gzip.open(filename, 'rb') as f:
#         f.read(16)  # skip the header
#         data = np.frombuffer(f.read(), dtype=np.uint8)
#         data = data.reshape(-1, 28, 28, 1)
#     return data
#
# def extract_labels(filename):
#     """Extract the labels into a 1D uint8 numpy array [index]."""
#     with gzip.open(filename, 'rb') as f:
#         f.read(8)  # skip the header
#         data = np.frombuffer(f.read(), dtype=np.uint8)
#     return data
#
# def load_mnist_from_files(batch_size=32, resize=(28, 28)):
#     # 文件路径
#     train_images_path = "/data/bigfiles/train-images-idx3-ubyte.gz"
#     train_labels_path = "/data/bigfiles/train-labels-idx1-ubyte.gz"
#     test_images_path = "/data/bigfiles/t10k-images-idx3-ubyte.gz"
#     test_labels_path = "/data/bigfiles/t10k-labels-idx1-ubyte.gz"
#
#     # 读取数据
#     train_images = extract_images(train_images_path)
#     train_labels = extract_labels(train_labels_path).astype(np.int32)
#     test_images = extract_images(test_images_path)
#     test_labels = extract_labels(test_labels_path).astype(np.int32)
#
#     # 将数据转换为MindSpore的数据集
#     train_dataset = ds.NumpySlicesDataset({"image": train_images, "label": train_labels}, shuffle=True)
#     test_dataset = ds.NumpySlicesDataset({"image": test_images, "label": test_labels}, shuffle=False)
#
#     # 定义数据增强操作
#     resize_op = vision.Resize(resize, interpolation=Inter.LINEAR)
#     rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
#     normalize_op = vision.Normalize(mean=[0.1307], std=[0.3081])
#     hwc2chw_op = vision.HWC2CHW()
#
#     # 将操作应用到数据集
#     train_dataset = train_dataset.map(operations=[resize_op, rescale_op, normalize_op, hwc2chw_op], input_columns="image")
#     test_dataset = test_dataset.map(operations=[resize_op, rescale_op, normalize_op, hwc2chw_op], input_columns="image")
#
#     # 批处理
#     train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
#     test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
#
#     return train_dataset, test_dataset
#
# # 加载数据集
# train_dataset, test_dataset = load_mnist_from_files(batch_size=32, resize=(28, 28))

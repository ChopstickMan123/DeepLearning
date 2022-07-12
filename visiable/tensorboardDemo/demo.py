# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net

from tensorboard import version

print(version.VERSION)
from torch.utils.tensorboard import SummaryWriter

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def main():
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # datasets 统一一下目录 ../../../../Data   ~/Data,~/GitCode/DeepLearning/visiable/tensorboardDemo
    trainset = torchvision.datasets.FashionMNIST('../../../../Data',
                                                 download=False,
                                                 train=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST('../../../../Data',
                                                download=False,
                                                train=False,
                                                transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # 定义网络
    net = Net()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ################ 1. TensorBoard setup
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    ################ 2. Writing to TensorBoard
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    ## 完了之后启动 tensorboard
    # tensorboard --logdir=runs
    # 注意要进入到项目目录下，runs是对应的输出目录 SummaryWriter('runs/fashion_mnist_experiment_1')


if __name__ == "__main__":
    main()

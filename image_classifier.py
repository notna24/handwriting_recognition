import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import os

import matplotlib.pyplot as plt

import time, os


data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),  #Crops the given image to random size and aspect ratio
        #transforms.RandomHorizontalFlip(), #randomly flips image horizontaly
        transforms.ToTensor(), #transforms input into tensor :D
        transforms.Normalize((0.1307,), (0.3081,)) #Normalizes a tensor image with mean and standard deviation
        #transforms.Normalize((0.15,), (0.3,))
    ])
}


download = False if os.path.isdir("data/MNIST") else True
train_dataset = datasets.MNIST('data',
                               train=True,
                               download=download,
                               transform=data_transforms['train']
                               )

#test_dataset = datasets.MNIST('data',
#                              train=False,
#                              download=download,
#                              transforms=data_transforms['val']
#                              )

training_data = torch.utils.data.DataLoader(train_dataset,
                                          shuffle=True,
                                          num_workers=0,
                                          drop_last=False,
                                          batch_size=64
                                          )

test_dataset = datasets.MNIST('data',
                               train=False,
                               download=download,
                               transform=data_transforms['train']
                               )

testing_data = torch.utils.data.DataLoader(train_dataset,
                                          shuffle=True,
                                          num_workers=0,
                                          drop_last=False,
                                          batch_size=64
                                          )

visualiziation_data = testing_data = torch.utils.data.DataLoader(train_dataset,
                                          shuffle=True,
                                          num_workers=0,
                                          drop_last=False,
                                          batch_size=1
                                          )


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.dropout_conv2 = nn.Dropout2d(p=0.4)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(250, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout_conv2(x)
        x = self.max_pool2(x)
        x = self.relu2(x)
        #print("here: ", x.size())
        x = x.view(-1, 250)
        #print("here2: ", x.size())
        x = self.fc1(x)
        x = self.soft_max(x)

        #print(x.size())
        return x






def imshow(inp, title=None, time=1):
    """Imshow for Tensor."""
    inp = inp.numpy()
    for i, image in enumerate(inp):
        #mean = np.array([0.1307])
        #std = np.array([0.3081])
        #inp = std * inp + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image[0])
        if title is not None:
            plt.title(title)#title[i].numpy())
        plt.pause(time)  # pause a bit so that plots are updated




def prep_target_data(target):
    #print(target)
    target = target.reshape((len(target), 1))
    new_target = np.array([np.array([0 if i != target[_] else 1 for i in range(10)], dtype='float32') for _ in range(len(target))])
    new_target = torch.tensor(new_target)
    #print(new_target)
    #print("new target: ", new_target.size())
    return new_target




def train(net, epoch=None):
    optimizer = optim.SGD(params=net.parameters(), lr=0.1, momentum=0.8)
    for batch, (inp, target) in enumerate(training_data):

        target = prep_target_data(target)

        target = Variable(target)
        inp = Variable(inp)


        optimizer.zero_grad()

        out = net(inp)

        criterion = F.mse_loss

        #print("out: ", out, "target: ", target)

        loss = criterion(out, target)

        #print("target: ", target[0], "out: ", out[0])
        print("epoch: ", epoch, "batch id: ", batch, "loss: ", loss)

        loss.backward()

        optimizer.step()
    torch.save(net, "models/Classifier_2.pt")


def test(net, epoch=None):
    for batch, (inp, target) in enumerate(testing_data):

        target = prep_target_data(target)

        target = Variable(target)
        inp = Variable(inp)

        out = net(inp)

        criterion = F.mse_loss

        #print("out: ", out, "target: ", target)

        loss = criterion(out, target)

        print("target: ", target[0], "out: ", out[0])
        print("test epoch: ", epoch, "batch id: ", batch, "loss: ", loss)



def visualize(net):
    for inp, target in visualiziation_data:
        Vtarget = prep_target_data(target)

        Vtarget = Variable(Vtarget)
        Vinp = Variable(inp)

        out = net(Vinp)

        t = target.numpy()[0]
        print(out.detach()[0].numpy()[0])
        print(out.detach()[0].numpy())
        o = out.detach()[0].numpy()
        prev = (0, 0)
        for i in range(10):
            print(o[i])
            if o[i] > prev[0]:
                prev = (o[i], i)

        criterion = F.mse_loss

        # print("out: ", out, "target: ", target)

        loss = criterion(out, Vtarget)

        imshow(Vinp, "target: " + str(t) + "\nout: " + str(prev[1]) + "\n" + str(prev[0] * 100) + "%", 1)

        print("target: ", target[0], "out: ", out[0])
        #print("test epoch: ", epoch, "batch id: ", batch, "loss: ", loss)
        print(target)
        print(torch.max(out))



if __name__ == "__main__":
    print("hello world")
    print(download)
    print(len(train_dataset))

    try:
        net = torch.load("models/Classifier_2.pt")
    except:
        net = Classifier()

    epochs = 100

    for epoch in range(epochs):
        #train(net, epoch)
        #test(net, epoch)
        visualize(net)

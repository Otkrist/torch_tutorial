from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

_cuda_ = torch.cuda.is_available()

print("Setting CUDA Availability to [%s]" % str(_cuda_))


data_train = MNIST('./pytorch_data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Scale((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./pytorch_data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Scale((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=32, num_workers=8)

net = LeNet5()
if _cuda_:
  net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        if _cuda_:
          images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
          images, labels = Variable(images), Variable(labels)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data[0])
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data[0]))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        if _cuda_:
          images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
          images, labels = Variable(images), Variable(labels)
        output = net(images)
        avg_loss += criterion(output, labels).sum(
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data[0], float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()

import matplotlib.pyplot as plt
import torchvision

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1,2,0))
    plt.show()

def display_result():
    data_test_loader = DataLoader(data_test, batch_size=4, num_workers=8, shuffle=False)
    images, labels =  next(iter(data_test_loader))
    #print labels
    if _cuda_:
      output = net(Variable(images.cuda()))
    else:
      output = net(Variable(images))
    pred = output.data.max(1)[1]
    print "Predicted OUTPUT: " , (pred.cpu().numpy())
    imshow(torchvision.utils.make_grid(images))

def main():
    for e in range(1, 2):
        train_and_test(e)
    display_result()


if __name__ == '__main__':
    main()

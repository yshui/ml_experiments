import torch
import torch.nn as nn
import sys
import numpy
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from PIL import Image
import numpy as np
from numpy.random import uniform as U
class Model:
    def __init__(self, n, size):
        super(Model, self).__init__()
        self.sz = size
        self.n = n
        loc = []
        # cov = []
        c= []
        for i in range(n):
            loc.append([U(0, size[0]), U(0, size[1])])
            # cov.append([U(1,10)])
            c.append([U(0, 1), U(0, 1), U(0, 1)])

        self.loc = torch.Tensor(loc).unsqueeze(1).cuda()
        self.loc.requires_grad = True
        # self.cov = torch.Tensor(cov).reshape(-1, 1, 1).cuda()
        # self.cov.requires_grad = True
        self.c = torch.Tensor(c).reshape(3, n, 1).cuda()
        self.c.requires_grad= True
        self.scale = torch.randn(n, 1, 1).cuda()
        self.scale.requires_grad = True
        indices = []
        for i in range(self.sz[0]):
            for j in range(self.sz[1]):
                indices.append([i,j])

        self.indices = torch.Tensor(indices).unsqueeze(0).cuda()
        self.indices.requires_grad = False
    def forward(self):
        dists = self.loc - self.indices
        dists = dists*self.scale.repeat(1, 1, 2)
        dists = torch.sum(dists*dists, dim=2).sqrt()
        v = 1 / dists
        return torch.sum(v.repeat(3, 1, 1) * self.c, dim=1).reshape(3, -1, self.sz[1])

    def parameters(self):
        return [self.loc, self.c]

print(torch.cuda.is_available())
pic = Image.open(sys.argv[1])
pic = torch.Tensor(numpy.array(pic.getdata()).reshape(pic.height, pic.width, 3))
pic = pic.transpose(0, 2).transpose(1, 2) / 255.0
pic = pic.cuda()
print("asd", pic.size())

m = Model(250, pic.size()[1:])
print(m.parameters())
optimizer = torch.optim.Adam(m.parameters(), lr=0.01, amsgrad=True)
t = 0
while True:
    pred = m.forward()
    loss = (pic-pred).pow(2).sum()
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 1000 == 0:
        im = pred.transpose(0, 2).transpose(0, 1).clamp(0,1)*255
        dd = Image.frombytes('RGB', (pic.size()[1], pic.size()[2]), bytes(bytearray(map(int, im.view(-1)))))
        print(dd.size)
        dd.show()
    t += 1

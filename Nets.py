import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from functional import reset_normal_param, LinearWeightNorm
# class Discriminator(nn.Module):
#     def __init__(self, output_units = 10):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 100)
#         self.fc2 = nn.Linear(100, output_units)

#     def forward(self, x, feature = False, cuda = False):
#         x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x_f = self.fc1(x)
#         x = F.leaky_relu(x_f)
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x if not feature else x_f

class Discriminator(nn.Module):
    def __init__(self, input_dim = 28 ** 2, output_dim = 10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 1000),
            LinearWeightNorm(1000, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        )
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)
        #for layer in self.layers:
        #    reset_normal_param(layer, 0.1)
        #reset_normal_param(self.final, 0.1, 5)
    def forward(self, x, feature = False, cuda = False):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0])
        if cuda:
            noise = noise.cuda()
        x = x + Variable(noise, requires_grad = False)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.relu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise.cuda()
            x = (x_f + Variable(noise, requires_grad = False))
        if feature:
            return x_f, self.final(x)
        return self.final(x)


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim = 28 ** 2):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias = False)
        self.bn1 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc2 = nn.Linear(500, 500, bias = False)
        self.bn2 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc3 = LinearWeightNorm(500, output_dim, weight_scale = 1)
        self.bn1_b = Parameter(torch.zeros(500))
        self.bn2_b = Parameter(torch.zeros(500))
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        #reset_normal_param(self.fc1, 0.1)
        #reset_normal_param(self.fc2, 0.1)
        #reset_normal_param(self.fc3, 0.1)
    def forward(self, batch_size, cuda = False):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False, volatile = not self.training)
        if cuda:
            x = x.cuda()
        x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.fc3(x))
        return x

#class Discriminator(nn.Module):
#    def __init__(self, nc = 1, ndf = 64, output_units = 10):
#        super(Discriminator, self).__init__()
#        self.ndf = ndf
#        self.main = nn.Sequential(
#            # state size. (nc) x 28 x 28
#            nn.Conv2d(nc, ndf, 4, 2, 3, bias=False),
#            nn.BatchNorm2d(ndf),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf) x 16 x 16
#            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*2) x 8 x 8
#            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*4) x 4 x 4
#            nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=False),
#        )
#        self.final = nn.Linear(ndf * 4, output_units, bias=False)
#    def forward(self, x, feature = False, cuda = False):
#        x_f = self.main(x).view(-1, self.ndf * 4)
#        return x_f if feature else self.final(x_f)

#class Generator(nn.Module):
#    def __init__(self, z_dim, ngf = 64, output_dim = 28 ** 2):
#        super(Generator, self).__init__()
#        self.z_dim = z_dim
#        self.main = nn.Sequential(
#            # input is Z, going into a convolution
#            nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
#            # state size. (ngf*8) x 4 x 4
#            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
#            # state size. (ngf*4) x 8 x 8
#            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
#            # state size. (ngf*2) x 16 x 16
#            nn.ConvTranspose2d(ngf, 1, 4, 2, 3, bias=False),
#            # state size. (ngf) x 32 x 32
#            nn.Sigmoid()
#        )
#    def forward(self, batch_size, cuda = False):
#        x = Variable(torch.rand(batch_size, self.z_dim, 1, 1), requires_grad = False, volatile = not self.training)
#        if cuda:
#            x = x.cuda()
#        return self.main(x)

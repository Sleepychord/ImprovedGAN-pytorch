import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from functional import reset_normal_param, LinearWeightNorm

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

    def forward(self, batch_size, cuda = False):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False, volatile = not self.training)
        if cuda:
            x = x.cuda()
        x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.fc3(x))
        return x

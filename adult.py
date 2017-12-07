import cPickle as pkl
import numpy as np
from torch.utils.data import TensorDataset
import torch
import random 
import argparse
from Nets import *
from ImprovedGAN import ImprovedGAN
from functional import normalize_infnorm
import pdb
class Adult(object):
    @staticmethod
    def onehot(attr_list, attr):
        assert attr in attr_list
        ret = [0] * len(attr_list)
        ret[attr_list.index(attr)] = 1
        return ret
    workclass_list = ['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked']
    marital_list = ['Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse']
    occupation_list = ['Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces']
    relationship_list = ['Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried']
    race_list = ['White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black']
    gender_list = ['Female','Male']
    country_list = ['United-States','Cambodia','England','Puerto-Rico','Canada','Germany','Outlying-US(Guam-USVI-etc)','India','Japan','Greece','South','China','Cuba','Iran','Honduras','Philippines','Italy','Poland','Jamaica','Vietnam','Mexico','Portugal','Ireland','France','Dominican-Republic','Laos','Ecuador','Taiwan','Haiti','Columbia','Hungary','Guatemala','Nicaragua','Scotland','Thailand','Yugoslavia','El-Salvador','Trinadad&Tobago','Peru','Hong','Holand-Netherlands']
    def __init__(self, read_file = '../data/adult'):
        if read_file is not None:
            self.positive_data = []
            self.negative_data = []
            f = open(read_file, 'r')
            for line in f.readlines():
                l = []
                t = line.split()
                if '?' in t:
                    continue
                l.append(int(t[0])) # age
                l.extend(Adult.onehot(Adult.workclass_list, t[1])) # workclass
                # l.append(int(t[2]))
                l.append(int(t[4])) # education-num
                l.extend(Adult.onehot(Adult.marital_list, t[5]))
                l.extend(Adult.onehot(Adult.occupation_list, t[6]))
                l.extend(Adult.onehot(Adult.relationship_list, t[7]))
                l.extend(Adult.onehot(Adult.race_list, t[8]))
                l.extend(Adult.onehot(Adult.gender_list, t[9]))
                l.append(int(t[10]))
                l.append(int(t[11]))
                l.append(int(t[12]))
                l.extend(Adult.onehot(Adult.country_list, t[13]))
                if t[14] == '>50K':
                    self.positive_data.append(l)
                else:
                    self.negative_data.append(l)
            else:
                pass
    def normalize(self):
        self.positive_data = normalize_infnorm(np.array(self.positive_data))
        self.negative_data = normalize_infnorm(np.array(self.negative_data))
    def LabeledData(self, class_num):
        data = torch.Tensor(np.concatenate((self.positive_data[:class_num], self.negative_data[:class_num])))
        label = torch.cat((torch.ones(class_num).long(), torch.zeros(class_num).long()))
        return TensorDataset(data, label)
    def UnlabeledData(self, balance = True):
        pos = self.positive_data[:-2000]
        neg = self.negative_data[:-2000]
        if balance:
            assert len(neg) >= len(pos)
            pos = np.tile(pos, ((len(neg)-1) // len(pos) + 1, 1))[:len(neg)]
        label = torch.cat((torch.ones(len(pos)).long(), torch.zeros(len(neg)).long()))
        data = torch.Tensor(np.concatenate((pos, neg)))
        return TensorDataset(data, label)
    def TestData(self):
        pos = self.positive_data[-2000:]
        neg = self.negative_data[-2000:]
        label = torch.cat((torch.ones(len(pos)).long(), torch.zeros(len(neg)).long()))
        data = torch.Tensor(np.concatenate((pos, neg)))
        return TensorDataset(data, label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Improved GAN')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
    parser.add_argument('--unlabel-weight', type=float, default=1, metavar='N',
                        help='scale factor between labeled and unlabeled data')
    parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH', help='logfile path, tensorboard format')
    parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH', help = 'saving path, pickle format')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    a = Adult()
    a.normalize()
    gan = ImprovedGAN(Generator(100, output_dim = 88), Discriminator(input_dim = 88, output_dim = 2), a.LabeledData(100), a.UnlabeledData(), a.TestData(), args)
    gan.train()

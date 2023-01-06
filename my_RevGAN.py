"""
This code is based AdvGAN. The generator has been replaced by RevGAN and required losses have been reconfigured.

"""

import os
import torch
from torch import nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import models
import torch.nn.functional as F
import torchvision
from itertools import chain
#################################################################################
#################################################################################
#################################################################################

models_path = './models/'

L1 = torch.nn.L1Loss()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class AdvGAN_Attack:
    def __init__(self,device,model,model_num_labels,image_nc,box_min,box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        
        self.gen_input_nc = image_nc
        self.netG = models.RevGanGenerator(image_nc).to(device)
        self.netDisc_X = models.Discriminator(image_nc).to(device)
        self.netDisc_Y = models.Discriminator(image_nc).to(device)
        # self.netG = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[0, 1])
        # self.netDisc_X = torch.nn.parallel.DistributedDataParallel(self.netDisc_X, device_ids=[0, 1])
        # self.netDisc_Y = torch.nn.parallel.DistributedDataParallel(self.netDisc_Y, device_ids=[0, 1])


        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc_X.apply(weights_init)
        self.netDisc_Y.apply(weights_init)

        # load checkpoint
        path = './'
        self.netG.load_state_dict(torch.load(path + 'netG_epoch_10.pth'))
        self.netDisc_X.load_state_dict(torch.load(path + 'netDisc_X_epoch_10.pth'))
        self.netDisc_Y.load_state_dict(torch.load(path + 'netDisc_Y_epoch_10.pth'))
        
        
        # initialize optimizers
        # self.optimizer_G = torch.optim.Adam(chain(self.netG.enc_X.parameters(), self.netG.dec_X.parameters()), lr=0.001)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D_X = torch.optim.Adam(self.netDisc_X.parameters(), lr=0.001)
        self.optimizer_D_Y = torch.optim.Adam(self.netDisc_Y.parameters(), lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels, epoch):
        # x_hat = self.netG.x_rec_only(x)
        # loss_x_rec= L1(x,x_hat)
        
        # if epoch < 1:
        #     # Optimize EncX,DecX only
        #     loss_x_rec.backward()
        #     self.optimizer_G.step()
        #     return {'loss_D':0, 'loss_cycle_X':0, 'loss_cycle_Y':0, 'loss_perturb':0, 'loss_adv':0, 'loss_x_rec':loss_x_rec.item()}

        # optimize D
        for i in range(1):
            ### LcGAN(F,Dy)
            y = self.netG.F(x)

            # WHY GIVE X AS REAL Y EXAMPLE ??????????
            self.optimizer_D_Y.zero_grad()
            pred_real = self.netDisc_Y(x)
            loss_D_X = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            # loss_D_X = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real,device=self.device))
            # loss_D_X.backward()

            pred_fake = self.netDisc_Y(y.detach())
            loss_D_Y = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            # loss_D_Y = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake,device=self.device))
            # loss_D_Y.backward()

            loss_D_GAN = loss_D_Y + loss_D_X
            loss_D_GAN.backward()
            self.optimizer_D_Y.step()

            ## LcGAN(G,Dx)
            x_hat = self.netG.G(y)

            self.optimizer_D_X.zero_grad()
            # pred_real = self.netDisc_X(y)
            pred_real = self.netDisc_X(x)
            loss_D_X = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            # loss_D_X = F.binary_cross_entropy(pred_real, torch.ones_like(pred_fake,device=self.device))
            # loss_D_X.backward()

            pred_fake = self.netDisc_X(x_hat.detach())
            loss_D_Y = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            # loss_D_Y = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake,device=self.device))
            # loss_D_Y.backward()

            loss_D_GAN = loss_D_Y + loss_D_X
            loss_D_GAN.backward()
            self.optimizer_D_X.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # y = self.netG.F(x)
            # x_hat = self.netG.G(y)
            # y_hat = self.netG.F(x_hat)

            #loss_cycle_X = L1(x, x_hat)
            loss_cycle_X = L1(x, x_hat)
            # loss_cycle_X.backward(retain_graph=True)

            # USE THIS LOSS OR NOT ???????
            loss_cycle_Y = L1(y,x)
            # loss_cycle_Y.backward(retain_graph=True)

            # WHY USE THIS LOSS ?????
            perturbation = y - x
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            C = 0.0001
            # C = torch.mean(torch.norm(x.view(x.shape[0], -1), 2, dim=1))
            loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(y)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            loss_adv = torch.sum(torch.max(real - other, torch.zeros_like(other)))

            # cal adv loss on reconstructed
            logits_model = self.model(x_hat)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            loss_class_rec = - torch.sum(torch.max(real - other, torch.zeros_like(other)))
            # Cross Entropy Loss
            # loss_class_rec = - F.cross_entropy(logits_model, labels)

            adv_lambda = .0001
            pert_lambda = .01
            class_rec_lambda = .0001
            # loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb + loss_cycle_X + loss_cycle_Y + class_rec_lambda * loss_class_rec
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb + loss_cycle_X + class_rec_lambda * loss_class_rec
            # loss_G = loss_cycle_X + class_rec_lambda * loss_class_rec
            
            loss_G.backward()
            self.optimizer_G.step()

        return {'loss_D':loss_D_GAN.item(),
                'loss_cycle_X':loss_cycle_X.item(),
                'loss_cycle_Y':loss_cycle_Y.item(),
                'loss_perturb':loss_perturb.item(),
                'loss_adv':loss_adv.item(),
                'loss_class_rec':loss_class_rec.item(),
                'loss_x_rec':0}

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            # if epoch == 1:
            #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.001)
            #     self.optimizer_D_X = torch.optim.Adam(self.netDisc_X.parameters(),lr=0.001)
            #     self.optimizer_D_Y = torch.optim.Adam(self.netDisc_Y.parameters(),lr=0.001)
            # if epoch == 150:
            #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.0001)
            #     self.optimizer_D_X = torch.optim.Adam(self.netDisc_X.parameters(),lr=0.0001)
            #     self.optimizer_D_Y = torch.optim.Adam(self.netDisc_Y.parameters(),lr=0.0001)
            # if epoch == 80:
            #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.00001)
            #     self.optimizer_D_X = torch.optim.Adam(self.netDisc_X.parameters(),lr=0.00001)
            #     self.optimizer_D_Y = torch.optim.Adam(self.netDisc_Y.parameters(),lr=0.00001)
            losses_sum = {'loss_D':0.0, 'loss_cycle_X':0.0, 'loss_cycle_Y':0.0, 'loss_perturb':0.0, 'loss_adv':0.0, 'loss_x_rec':0.0,'loss_class_rec':0.0}
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                losses_batch = self.train_batch(images, labels, epoch)
                for k in losses_batch.keys():
                    losses_sum[k] += losses_batch[k]

            # save generator
            if epoch%5==0:
                torch.save(self.netG.state_dict(), models_path + 'netG_epoch_' + str(epoch) + '.pth')
                torch.save(self.netDisc_X.state_dict(), models_path + 'netDisc_X_epoch_' + str(epoch) + '.pth')
                torch.save(self.netDisc_Y.state_dict(), models_path + 'netDisc_Y_epoch_' + str(epoch) + '.pth')

            # print statistics
            num_batch = len(train_dataloader)
            print('epoch ', epoch)
            for k,v in losses_sum.items():
                print("\t%s\t%.3f" % (k, v/num_batch))

#################################################################################
#################################################################################
#################################################################################
#################################################################################

use_cuda = True
image_nc = 1
model_num_labels = 10
epochs = 100
batch_size = 512
BOX_MIN = 0
BOX_MAX = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
pretrained_model = "./MNIST_target_model.pth"
targeted_model = models.MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model,map_location=device))
targeted_model.eval()


# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=40)

advGAN = AdvGAN_Attack(device,targeted_model,model_num_labels,image_nc,BOX_MIN,BOX_MAX)
advGAN.train(dataloader, epochs)
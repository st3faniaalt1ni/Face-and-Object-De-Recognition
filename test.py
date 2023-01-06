import torch
import models
import matplotlib.pyplot as plt
from pandas import read_csv
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision

batch_size = 64


data_test = torchvision.datasets.MNIST('./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(data_test,shuffle=False,batch_size=batch_size)

s = next(iter(dataloader))
x = s[0]
labels_test = s[1]


import sys
model = models.RevGanGenerator(1)
model.load_state_dict(torch.load('models/netG_epoch_'+sys.argv[1]+'.pth'))
# model.load_state_dict(torch.load('./netG_epoch_10.pth'))
model.eval()

# Generate/Reconstruct
y = model.F(x)
x_hat = model.G(y)

MSELoss = torch.nn.MSELoss()

# MSE(x,x_rec)
mse_loss = MSELoss(x,x_hat).item()
print('MSE(x,x_rec): ' ,mse_loss)

# MSE(x,y_adv)
mse_loss2 = MSELoss(x,y).item()
print('MSE(x,y_adv): ' ,mse_loss2)

from skimage.metrics import structural_similarity as ssim

# SSIM(x,x_rec)
im1 = x.detach().cpu().numpy()
im2 = x_hat.detach().cpu().numpy()
ssim_loss = 0.0
for i in range(batch_size):
    tmp_ssim = ssim(im1[i].squeeze(),im2[i].squeeze())
    ssim_loss += tmp_ssim/batch_size
print('SSIM(x,x_rec): ', ssim_loss)

# SSIM(x,y_adv)
im1 = x.detach().cpu().numpy()
im2 = y.detach().cpu().numpy()
ssim_loss2 = 0.0
for i in range(batch_size):
    tmp_ssim = ssim(im1[i].squeeze(),im2[i].squeeze())
    ssim_loss2 += tmp_ssim/batch_size
print('SSIM(x,y_adv): ', ssim_loss2)

# Accuracy
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

classifier = models.MNIST_target_net().to(device)
classifier.load_state_dict(torch.load("MNIST_target_model.pth",map_location=device))
classifier.eval()

logits = classifier(x.cuda())
probs = F.softmax(logits, dim=1)
preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
print('Fooling(x): ', (1 - sum([p == l for p,l in zip(preds,labels_test)])/len(labels_test)).item())

logits = classifier(y.cuda())
probs = F.softmax(logits, dim=1)
preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
print('Fooling(adv): ', (1 - sum([p == l for p,l in zip(preds,labels_test)])/len(labels_test)).item())

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1, (10., 10.))
grid = ImageGrid(fig, 111,nrows_ncols=(8, 8),axes_pad=0.4,)
for i,axes in enumerate(grid):
    axes.set_title(preds[i], fontdict=None, loc='center', color = "k")
    axes.imshow(y.view(batch_size, 1, 28, 28)[i].squeeze().detach().cpu().numpy(),cmap='gray')
    axes.axis('off')
plt.savefig('classifications.png')

logits = classifier(x_hat.cuda())
probs = F.softmax(logits, dim=1)
preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
print('Fooling(x_rec): ', (1 - sum([p == l for p,l in zip(preds,labels_test)])/len(labels_test)).item())

save_image(x.view(batch_size, 1, 28, 28),'./original.png')
save_image(y.view(batch_size, 1, 28, 28),'./adversarial.png')
save_image(x_hat.view(batch_size, 1, 28, 28),'./reconstructed.png')



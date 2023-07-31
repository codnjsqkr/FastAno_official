import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, batch_size, num_of_img=3):
        super(Generator, self).__init__()
        bn = None
        if batch_size == 1:
            bn = False 
        else:
            bn = True
        self.conv1 = nn.Conv3d(num_of_img, 64, 3, 2, 1)
        conv2 = [nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 3, 2, 1)]
        if bn == True:
            conv2 += [nn.BatchNorm3d(128)]
        else:
            conv2 += [nn.InstanceNorm3d(128)]
        self.conv2 = nn.Sequential(*conv2)
        conv3 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv3d(128, 256, 3, 2, 1)]
        if bn == True:
            conv3 += [nn.BatchNorm3d(256)]
        else:
            conv3 += [nn.InstanceNorm3d(256)]
        self.conv3 = nn.Sequential(*conv3)
        deconv3 = [nn.ReLU(),
                   nn.ConvTranspose3d(128 * 2, 128, (2, 2, 2), 2, 0)]
        if bn == True:
            deconv3 += [nn.BatchNorm3d(128)]
        else:
            deconv3 += [nn.InstanceNorm3d(128)]
        self.deconv3 = nn.Sequential(*deconv3)
        deconv2 = [nn.ReLU(),
                   nn.ConvTranspose3d(128 * 2, 64, (3, 4, 4), 2, 1)]
        if bn == True:
            deconv2 += [nn.BatchNorm3d(64)]
        else:
            deconv2 += [nn.InstanceNorm3d(64)]
        self.deconv2 = nn.Sequential(*deconv2)
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(64 * 2, 1, (1, 2, 2), (1, 2, 2), (1, 0, 0)), 
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        d2 = self.deconv3(c3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1)
        out = self.deconv1(d1)
        return out

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion(nn.Module):
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=.02,img_size=64,device="cuda"):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        beta = self.prepare_noise_schedule().to(device)
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha,dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)

    def prepare_noise_schedule(self):
        ## Create a linspace using the torch linspace
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return betas
    
    def noise_images(self, x, t):
        ## Generate our sqrt alpha hat and sqrt one minus alpha hat, and nosie the image
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # basically "Unsqueezes" the tensor
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    
    # Sample random timesteps for training
    def sample_timesteps(self,n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, x, timestep):
        t = (torch.ones(x.size(0)) * timestep).long().to(self.device)
        predicted_noise = model(x, t)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        if timestep > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

diffuser = Diffusion(device="cpu")
print("Stopping")

# class UNet(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8)

#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 256)
#         self.sa4 = SelfAttention(128, 16)
#         self.up2 = Up(256, 128)
#         self.sa5 = SelfAttention(64, 32)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        

# diffuser = Diffusion(noise_steps=10,device="cpu")
# print("Stopping")
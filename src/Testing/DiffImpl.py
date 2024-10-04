import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=.02,img_size=64,device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

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
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            pass
    

diffuser = Diffusion(noise_steps=10,device="cpu")
print("Stopping")
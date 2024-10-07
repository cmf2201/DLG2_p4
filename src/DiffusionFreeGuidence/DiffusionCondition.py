
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        betas = torch.linspace(beta_1, beta_T, T)  
        self.register_buffer('betas', betas )
        
        # Get alphas and cumprods of alphas

        alphas = 1-self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1-alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', sqrt_alphas_bar )
        self.register_buffer('sqrt_one_minus_alphas_bar', sqrt_one_minus_alphas_bar )

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        
        # Get batch size and image shape
        batch_size = x_0.shape[0]
        x_shape = x_0.shape 
        # Pick batched random timesteps (torch.Size([batch_size]))
        rand_t = torch.randint(low=1, high=self.T, size=(batch_size,)).to_device(self.device)  
        # Generate random noise from normal distribution (torch.Size([batch_size, 3, 32, 32]))
        actual_noise = torch.randn_like(x_0).to_device(self.device)
        # Extract coefficients for the current timesteps
        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, rand_t, x_shape) 
        sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, rand_t, x_shape) 
        # Compute x_t for all batch elements
        x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * actual_noise
        # Call your diffusion model to predict the noise
        predicted_noise = self.model(x_t, rand_t, labels) 
        # Compute the loss between predicted noise and actual noise
        lossfcn = torch.nn.MSELoss()
        loss = lossfcn(predicted_noise, actual_noise)
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w
        self.beta_1 = beta_1
        self.beta_T = beta_T
        betas = torch.linspace(beta_1, beta_T, T)          
        alphas = 1-self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas', alphas )
        self.register_buffer('alphas_bar', alphas_bar )
        self.register_buffer('betas', betas )

        # Store any constant in register_buffer for quick usage in forward step


    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            x_t = torch.randn_like(labels)
            predicted_noise = self.model(x_t, time_step)
            alpha = extract(self.alphas, x_t, labels)
            alpha_bar = extract(self.alphas_bar, x_t, labels)
            beta = extract(self.beta, x_t, labels)
            if(time_step > 1):
                z = torch.randn_like(labels).cuda()
            else:
                z = torch.zeros_like(labels).cuda()
            x_t = 1/torch.sqrt(alpha) * (x_t - ((1-alpha)) / (torch.sqrt(1-alpha_bar)) * predicted_noise) + torch.sqrt(beta) * z            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)    



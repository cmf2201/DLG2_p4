
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from Testing.DiffImpl import Diffusion


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,device="cuda"):
        super().__init__()

        self.model = model
        self.T = T
        self.device = device
        
        # YOUR IMPLEMENTATION HERE!
        self.diffuser = Diffusion(noise_steps=T, beta_start=beta_1, beta_end=beta_T, img_size=32, device=device)
        # Precompute and store the parameters for performing noise addition for a given timestep.
        

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        
        # pick batched random timestep below self.T. (torch.Size([batch_size]))
        t = self.diffuser.sample_timesteps(x_0.shape[0]).to(self.device)
        # Generate random noise from normal distribution with 0 mean and 1 variance (torch.Size([batch_size, 3, 32, 32])
        # Compute the x_t (images obtained after corrupting the input images by t times)  (torch.Size([batch_size, 3, 32, 32])
        x_t, noise = self.diffuser.noise_images(x_0, t)
        # Call your diffusion model to get the predict the noise -  t is a random index
        # self.model(x_t, t, labels)
        predicted_noise = self.model(x_t, t, labels)
        # Compute your loss for model prediction and ground truth noise (that you just generated)
        mse = nn.MSELoss()
        loss = mse(noise,predicted_noise)
    
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.,device="cuda"):
        super().__init__()

        self.model = model
        self.T = T
        self.diffuser = Diffusion(noise_steps=T, beta_start=beta_1, beta_end=beta_T, img_size=32, device=device)
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w


        # YOUR IMPLEMENTATION HERE!

        # Store any constant in register_buffer for quick usage in forward step


    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            with torch.no_grad():
                print(time_step)

                # YOUR IMPLEMENTATION HERE!
                self.diffuser.sample(self.model,x_t)
                
                
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   



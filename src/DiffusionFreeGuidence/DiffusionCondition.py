
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import os

import numpy as np

to_image = ToPILImage()

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def iterative_process(input, normal, beta, sqrt_alpha):
    # input [3x32x32]
    # output [3x32x32]
    # normal = [3x32x32]
    # beta = [1]
    # sqrt_alpha = [1]

    varience = beta * torch.eye(32).cuda()

    std = torch.square(varience).cuda()
    mean = input * sqrt_alpha

    output = normal * std + mean

    del varience, std, mean
    return output

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        is_lin = True
        s = 1e-3
        self.model = model
        self.T = T
        
        # YOUR IMPLEMENTATION HERE!
        if is_lin:
            betas_tensor = torch.linspace(beta_1, beta_T, T + 1)
            alphas = 1 - betas_tensor
            print(alphas)
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            print(alphas_cumprod)
            sqrt_alphas_tensor = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_tensor = torch.sqrt(1 - alphas_cumprod)
        else:
            range_tensor = torch.linspace(beta_1, beta_T, T + 1)

            alpha_t = torch.cos((range_tensor/T + s)/(1 + s) * (torch.pi / 2)) ** 2
            alpha_0 = alpha_t[0]

            alphas_cumprod = alpha_t / alpha_0
            betas_tensor = torch.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
            sqrt_alphas_tensor = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_tensor = torch.sqrt(1 - alphas_cumprod)
        
        # Precompute and store the parameters for performing noise addition for a given timestep. CHECK
        
        # Get the betas in beta_1, beta_T range CHECK
        self.register_buffer('betas', betas_tensor)
        
        # Get alphas and cumprods of alphas CHECK

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', sqrt_alphas_tensor)
        self.register_buffer('sqrt_one_minus_alphas_bar', sqrt_one_minus_alphas_tensor)

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        batch_size = x_0.size(dim=0)

        # pick batched random timestep below self.T. (torch.Size([batch_size]))
        timesteps = torch.randint(low=1, high=(self.T + 1), size=(1,batch_size))[0]

        # Generate random noise from normal distribution with 0 mean and 1 variance (torch.Size([batch_size, 3, 32, 32])
        zeros = torch.zeros(size=(batch_size, 3, 32, 32))
        ones = torch.ones(size=(batch_size, 3, 32, 32))
        normal_distributions = torch.normal(zeros, ones).cuda()
        # Compute the x_t (images obtained after corrupting the input images by t times)  (torch.Size([batch_size, 3, 32, 32])
        """for loop or no? is there a torch function"""
        x_t_list = []
        for batch_i in range(batch_size):
            # get important values for equation
            t = timesteps[batch_i]
            orig_image = x_0[batch_i]
            normal = normal_distributions[batch_i]
            betas = self.betas[:t]
            sqrt_alphas = self.sqrt_alphas_bar[:t]
            
            x_t_minus_1 = orig_image

            for iter in range(t):
                beta = betas[iter]
                sqrt_alpha = sqrt_alphas[iter]
                x_t_minus_1 = iterative_process(x_t_minus_1, normal, beta, sqrt_alpha) # NOT WORKING YET
            x_t_list.append(x_t_minus_1.detach()) # No gradients for this right? this is ground truth?
        
        x_t = torch.stack(x_t_list).cuda()

        # Call your diffusion model to get the predict the noise -  t is a random index
        # self.model(x_t, t, labels)
        predict = self.model(x_t.cuda(), timesteps.cuda(), labels.cuda())

        for batch_i in range(batch_size):
            x_0_bi = x_0[batch_i]
            x_t_bi = x_t[batch_i]
            pred_bi = predict[batch_i]

            x_0_img = to_image(x_0_bi)
            x_t_img = to_image(x_t_bi)
            pred_img = to_image(pred_bi)

            x_0_img.save('ImgOutputs/x_0-' + str(batch_i) + '.png')
            x_t_img.save('ImgOutputs/x_t-' + str(batch_i) + '.png')
            pred_img.save('ImgOutputs/pred-' + str(batch_i) + '.png')

        
        # Compute your loss for model prediction and ground truth noise (that you just generated)
        loss = None
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


        # YOUR IMPLEMENTATION HERE!

        # Store any constant in register_buffer for quick usage in forward step


    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)

            # YOUR IMPLEMENTATION HERE!
            
            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   



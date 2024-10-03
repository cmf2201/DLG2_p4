import math
import torch
#linears
beta_1 = 1e-4
beta_T = 0.028
T = 500

lin_betas_tensor = torch.linspace(beta_1, beta_T, T + 1)

#cosine
beta_1 = 0
beta_T = T
T = 500
s = 1e-3
batch_size = 4

cos_range_tensor = torch.linspace(beta_1, beta_T, T + 1)

cos_alpha_t = torch.cos((cos_range_tensor/T + s)/(1 + s) * (torch.pi / 2)) ** 2
cos_alpha_0 = cos_alpha_t[0]

cos_alpha = cos_alpha_t / cos_alpha_0
cos_betas_tensor = torch.clip(1 - cos_alpha[1:] / cos_alpha[:-1], 0, 0.999)

zeros = torch.zeros(size=(batch_size, 3, 32, 32))
ones = torch.ones(size=(batch_size, 3, 32, 32))
normal = torch.normal(zeros, ones)

timesteps = torch.randint(low=0, high=(T + 1), size=(1,batch_size))[0]

print(timesteps)
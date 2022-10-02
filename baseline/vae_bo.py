import torch
import torch.nn as nn
from torch.nn import functional as F
import botorch
import numpy as np
import pandas as pd
import random
import argparse
from benchmark import get_problem
from baseline.vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf
from utils import latin_hypercube, from_unit_cube, save_results, save_args


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        
        self.latent_mapping = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        latent_z = self.latent_mapping(z)
        out = self.decoder(latent_z)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
    
    
def train_vae(train_x, epochs=30):
    train_x = torch.tensor(train_x, dtype=torch.float)
    for epoch in range(epochs):
        for idx in range(train_x.shape[0]):
            x = train_x[idx: idx+1]
            out_x, mu, logvar = vae_model(x)
            recons_loss = F.mse_loss(out_x, x)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
            loss = recons_loss + 0.5 * kld_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        # print(loss)


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--init_samples', default=10, type=int)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--update_interval', default=20, type=int)
parser.add_argument('--active_dims', default=6, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': 'vae_{}'.format(args.active_dims),
    'func': args.func,
    'seed': args.seed
}
func = get_problem(args.func, save_config)
dims = func.dims
lb = func.lb
ub = func.ub

save_args(
    'config/' + args.root_dir,
    'vae_{}'.format(args.active_dims),
    args.func,
    args.seed,
    args
)

vae_model = VAE(func.dims, args.active_dims)
opt = torch.optim.Adam(vae_model.parameters(), lr=args.lr)

# train_x, train_y = generate_initial_data(func, args.init_samples, lb, ub)
points = latin_hypercube(args.init_samples, dims)
points = from_unit_cube(points, lb, ub)
train_x, train_y = [], []
for i in range(args.init_samples):
    y = func(points[i])
    train_x.append(points[i])
    train_y.append(y)
sample_counter = args.init_samples
best_y  = [(sample_counter, np.max(train_y))]

train_vae(train_x)

while True:
    if sample_counter % args.update_interval == 0:
        train_vae(train_x)
    
    np_train_y = np.array(train_y)
    mu, logvar = vae_model.encode(torch.tensor(train_x, dtype=torch.float))
    z = vae_model.sample_z(mu, logvar)
    z = z.detach().numpy()
    z = np.clip(z, lb[0], ub[0])
    
    gpr = get_gpr_model()
    gpr.fit(z, np_train_y)
    new_z, _ = optimize_acqf(args.active_dims, gpr, z, np_train_y, args.batch_size, lb[0]*args.active_dims, ub[0]*args.active_dims)
    new_x = vae_model.decode(torch.tensor(new_z, dtype=torch.float))
    new_x = new_x.detach().numpy()
    
    new_x = [np.clip(new_x[i], lb[0], ub[0]) for i in range(new_x.shape[0])]
    new_y = [func(x) for x in new_x]
    
    train_x.extend(new_x)
    train_y.extend(new_y)
    sample_counter += len(new_x)
    if sample_counter >= args.max_samples:
        break
        
print('best f(x):', func.tracker.best_value_trace[-1])
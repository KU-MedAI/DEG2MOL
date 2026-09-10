import torch
import torch.nn as nn
import torch.nn.functional as F

class GO_Encoder(nn.Module):
    def __init__(self, dims, latent_dim=256):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
            
        self.bottleneck = nn.Linear(dims[-1], latent_dim)
        
    def forward(self, x):
        h = x
        for layer, bn in zip(self.layers, self.bns):
            h = F.mish(bn(layer(h)))
        return self.bottleneck(h)

class GO_Decoder(nn.Module):
    def __init__(self, dims, latent_dim=256):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        rev_dims = dims[::-1] 
        
        self.input_layer = nn.Linear(latent_dim, rev_dims[0])
        self.input_bn = nn.BatchNorm1d(rev_dims[0])
        
        for i in range(len(rev_dims) - 1):
            self.layers.append(nn.Linear(rev_dims[i], rev_dims[i+1]))
            if i < len(rev_dims) - 2: 
                self.bns.append(nn.BatchNorm1d(rev_dims[i+1]))
                
    def forward(self, z):
        h = F.mish(self.input_bn(self.input_layer(z)))
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.mish(self.bns[i](h))
        return h


class GO_Autoencoder(nn.Module):
    def __init__(self, dims, latent_dim=256):
        super().__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        
        self.encoder = GO_Encoder(dims, latent_dim)
        self.decoder = GO_Decoder(dims, latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z, None, None

    def reparameterize(self, mu, logvar):
        return mu

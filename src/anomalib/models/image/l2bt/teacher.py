import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.fe = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.layers = layers

        self.patch_size = self.fe.patch_size
        self.embed_dim = self.fe.embed_dim

    def forward(self, x):
        return self.fe.get_intermediate_layers(x, self.layers)

    def forward_last(self, x):
        return self.fe.forward_features(x)['x_norm_patchtokens'].view(-1, self.embed_dim)

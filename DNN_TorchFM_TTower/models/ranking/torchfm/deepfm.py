# DNN_TorchFM_TTower\models\ranking\torchfm\deepfm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)

    def forward(self, x):
        return self.embedding(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, dims, dropout):
        super().__init__()
        layers = []
        for dim in dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mlp = MultiLayerPerceptron(len(field_dims) * embed_dim, mlp_dims, dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + torch.sum(self.mlp(embed_x.view(embed_x.size(0), -1)), dim=1, keepdim=True)
        return torch.sigmoid(x.squeeze(1))


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

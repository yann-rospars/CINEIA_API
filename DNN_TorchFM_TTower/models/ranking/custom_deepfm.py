"""
纯 PyTorch 实现的 DeepFM
支持:
  • sparse_x : LongTensor (batch, num_sparse_fields)
  • dense_x  : FloatTensor (batch, num_dense_fields)
输出 raw logit (BCEWithLogitsLoss 对应)
"""

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """ y = Σ w_i + b  （一阶项） """
    def __init__(self, field_dims):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x.shape = (B, F)  每个元素都是“在其 field 的全局偏移 id”
        return self.fc(x).sum(dim=1) + self.bias


class DenseLinear(nn.Module):
    """ 一阶 dense： y = w·x """
    def __init__(self, num_dense):
        super().__init__()
        self.fc = nn.Linear(num_dense, 1, bias=False)

    def forward(self, x):
        return self.fc(x)


class FeaturesEmbedding(nn.Module):
    """ 二阶 / Deep 部分公用的 embedding """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)

    def forward(self, x):
        return self.embedding(x)          # (B, F, D)


class FactorizationMachine(nn.Module):
    """ FM 二阶交叉项 Σ⟨v_i, v_j⟩ """
    def forward(self, embed_x):
        # embed_x : (B, F, D)
        square_of_sum = embed_x.sum(dim=1) ** 2        # (B, D)
        sum_of_square = (embed_x ** 2).sum(dim=1)      # (B, D)
        ix = 0.5 * (square_of_sum - sum_of_square)     # (B, D)
        return ix.sum(dim=1, keepdim=True)             # (B, 1)


class MLP(nn.Module):
    def __init__(self, in_dim, dims, dropout):
        super().__init__()
        layers = []
        for dim in dims:
            layers += [nn.Linear(in_dim, dim),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)       # (B, last_dim)


class DeepFM(nn.Module):
    def __init__(self,
                 field_dims,          # List[int] sparse vocab_sizes
                 num_dense,           # int
                 embed_dim=16,
                 mlp_dims=(128, 64),
                 dropout=0.2):
        super().__init__()
        self.num_dense = num_dense

        self.linear_sparse = FeaturesLinear(field_dims)
        self.linear_dense  = DenseLinear(num_dense)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine()

        dnn_input_dim = len(field_dims) * embed_dim + num_dense
        self.mlp = MLP(dnn_input_dim, mlp_dims, dropout)
        self.mlp_out = nn.Linear(mlp_dims[-1], 1)

    def forward(self, sparse_x, dense_x):
        """
        sparse_x : LongTensor (B, F_s)
        dense_x  : FloatTensor(B, F_d)
        """
        embed_x = self.embedding(sparse_x)           # (B, Fs, D)

        linear_term = self.linear_sparse(sparse_x) + self.linear_dense(dense_x)
        fm_term     = self.fm(embed_x)               # (B, 1)

        dnn_input   = torch.cat([embed_x.reshape(embed_x.size(0), -1),
                                 dense_x], dim=1)
        dnn_term    = self.mlp_out(self.mlp(dnn_input))  # (B, 1)

        logit = linear_term + fm_term + dnn_term     # (B, 1)
        return logit.squeeze(1)                      # (B,)

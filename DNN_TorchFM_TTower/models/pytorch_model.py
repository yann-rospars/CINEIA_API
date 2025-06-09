import torch
import torch.nn as nn

class TwoTowerMLPModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, hidden_dim=64):
        super(TwoTowerMLPModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)  # logit 

    def forward(self, user_ids, movie_ids):
        user_vec = self.user_embedding(user_ids)     # (batch_size, embedding_dim)
        movie_vec = self.movie_embedding(movie_ids)  # (batch_size, embedding_dim)

        x = torch.cat([user_vec, movie_vec], dim=1)  # (batch_size, embedding_dim*2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logit = self.fc2(x).squeeze(1)              # (batch_size,)
        return logit  # raw logits (BCEWithLogitsLoss)

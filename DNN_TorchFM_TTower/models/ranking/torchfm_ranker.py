# DNN_TorchFM_TTower\models\ranking\torchfm_ranker.py
"""
torch-fm 版 DeepFM 精排模型封装
可被 train_ranking / infer_ranking 调用
"""

from pathlib import Path
from typing import Sequence

import torch
# from torchfm.deepfm import DeepFM
from models.ranking.torchfm.deepfm import DeepFM

MODEL_PATH = Path("saved_model/deepfm_ranker.pt")


def create_model(field_dims: Sequence[int],
                 embed_dim: int = 16):
    """
    field_dims : List[int]  每个 sparse 特征（user_id, movie_id, genre_id）的 vocab_size
    """
    model = DeepFM(field_dims,
                   embed_dim=embed_dim,
                   mlp_dims=(128, 64),
                   dropout=0.2)
    return model


def save_model(model: torch.nn.Module) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


def load_model(field_dims: Sequence[int]):
    """
    若模型文件存在则加载并返回 eval() 后模型，否则返回 None
    """
    if MODEL_PATH.exists():
        m = create_model(field_dims)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        return m
    return None

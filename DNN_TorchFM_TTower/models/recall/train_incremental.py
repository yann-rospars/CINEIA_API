# models/recall/train_incremental.py
"""
Two-Tower 模型增量训练脚本
= 原 train_incremental.py，但从 train_two_tower 导入数据构造工具
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.db import fetchone_dict
from models.pytorch_model import TwoTowerMLPModel
from models.recall.train_two_tower import (
    generate_training_data,
    RecommendationDataset,
)

MODEL_PATH = "saved_model/dnn_recommender.pt"


def incremental_train(neg_ratio: int = 1,
                      epochs: int = 3,
                      lr: float = 5e-4):
    """
    • 用最新行为重新构造正/负样本  
    • 在已有模型参数基础上小步训练
    """
    df = generate_training_data(neg_ratio)
    if df.empty:
        print("[incremental] 无训练数据，跳过")
        return

    loader = DataLoader(RecommendationDataset(df),
                        batch_size=64, shuffle=True)

    max_u = fetchone_dict("SELECT MAX(id) AS m FROM users")["m"] or 0
    max_m = fetchone_dict("SELECT MAX(id) AS m FROM movies")["m"] or 0

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "[incremental] 基础模型不存在，请先完整训练"
        )

    model = TwoTowerMLPModel(max_u, max_m,
                             embedding_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    tic = time.time()
    for ep in range(1, epochs + 1):
        losses = []
        for u, m, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(u, m), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[incremental] epoch {ep}/{epochs}  "
              f"loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[incremental] Δ 训练完成，用时 {time.time() - tic:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg_ratio", type=int, default=1)
    args = ap.parse_args()

    incremental_train(neg_ratio=args.neg_ratio, epochs=args.epochs)

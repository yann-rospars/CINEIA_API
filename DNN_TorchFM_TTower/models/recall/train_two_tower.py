# models/recall/train_two_tower.py
"""
Two-Tower 召回模型离线训练脚本
= 旧版 train.py，调整了 import 路径并改名
"""
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # ← 新增

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.db import fetchall_dict, fetchone_dict
from models.pytorch_model import TwoTowerMLPModel

# ---------------------------------------------------------------------------
#                     固定 saved_model 目录到包根下                           #
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[2]           # …/CINEIA/DNN_TorchFM_TTower
SAVE_DIR   = ROOT_DIR / "saved_model"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "dnn_recommender.pt"


# os.makedirs("saved_model", exist_ok=True)

# --------------------------------------------------------------------------- #
#                            数据提取 / 特征构造                               #
# --------------------------------------------------------------------------- #
def get_all_movies() -> List[int]:
    return [r["id"] for r in fetchall_dict("SELECT id FROM movies")]


def get_movie_genres() -> Dict[int, Set[int]]:
    rows = fetchall_dict("SELECT movie_id, genre_id FROM movie_genre")
    m2g = {}
    for r in rows:
        m2g.setdefault(r["movie_id"], set()).add(r["genre_id"])
    return m2g


def get_positive_samples() -> pd.DataFrame:
    """
    view_history 中的每条观看记录视为正样本 (label=1)
    """
    rows = fetchall_dict("SELECT user_id, movie_id FROM view_history")
    if not rows:
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    df = pd.DataFrame(rows)
    df["rating"] = 1.0
    return df


def generate_training_data(neg_ratio: int = 1) -> pd.DataFrame:
    """
    正样本：view_history  
    负样本：同用户未看过，且题材与已看影片不重叠的随机采样
    """
    pos_df = get_positive_samples()
    if pos_df.empty:
        return pos_df

    all_movies = set(get_all_movies())
    movie_genres = get_movie_genres()

    user_pos = pos_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    user_genres = {
        u: {g for m in ms for g in movie_genres.get(m, set())}
        for u, ms in user_pos.items()
    }

    neg_records = []
    for u, pos in user_pos.items():
        watched_genres = user_genres.get(u, set())
        # candidate: 未看 & 题材不重叠
        cand = [m for m in all_movies - pos
                if movie_genres.get(m, set()).isdisjoint(watched_genres)]
        num_neg = min(len(cand), neg_ratio * len(pos))
        if num_neg:
            neg_records += [{"user_id": u, "movie_id": m, "rating": 0.0}
                            for m in random.sample(cand, num_neg)]

    neg_df = pd.DataFrame(neg_records)
    train_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return train_df


class RecommendationDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


def _get_max_ids() -> tuple[int, int]:
    mu = fetchone_dict("SELECT MAX(id) AS m FROM users")["m"] or 0
    mm = fetchone_dict("SELECT MAX(id) AS m FROM movies")["m"] or 0
    return mu, mm


# --------------------------------------------------------------------------- #
#                               训 练 主 程 序                                 #
# --------------------------------------------------------------------------- #
def main(epochs: int = 3, batch_size: int = 128, neg_ratio: int = 1):
    df = generate_training_data(neg_ratio)
    if df.empty:
        print("[train_two_tower] ❌ 训练集为空")
        return

    print(f"[train_two_tower] 样本 {len(df)} | 用户 {df['user_id'].nunique()} | 电影 {df['movie_id'].nunique()}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(RecommendationDataset(train_df),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RecommendationDataset(val_df),
                            batch_size=batch_size, shuffle=False)

    max_u, max_m = _get_max_ids()
    model = TwoTowerMLPModel(max_u, max_m, embedding_dim=32, hidden_dim=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        tloss = []

        for u, m, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=80):
            u, m, y = u.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(u, m), y)
            loss.backward()
            optimizer.step()
            tloss.append(loss.item())

        avg_t = np.mean(tloss)

        # ---------- 验证 ----------
        model.eval()
        vloss = []
        with torch.no_grad():
            for u, m, y in val_loader:
                u, m, y = u.to(device), m.to(device), y.to(device)
                vloss.append(criterion(model(u, m), y).item())
        avg_v = np.mean(vloss)

        print(f"[Ep {ep:02}] train={avg_t:.4f}  val={avg_v:.4f}  time={time.time()-ep_start:.1f}s")

        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), MODEL_PATH)
            print("   ↳  Model saved")

    print(f"[train_two_tower] Done，best val={best_val:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)        # 默认为 3
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--neg_ratio", type=int, default=1)
    args = ap.parse_args()

    main(epochs=args.epochs, batch_size=args.batch, neg_ratio=args.neg_ratio)
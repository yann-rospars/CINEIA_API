"""
训练自定义 DeepFM 精排模型（统一 4 维稠密特征：
  recall_score · vote_average · popularity · age）
运行:
    python -m models.ranking.train_ranking --epochs 3
"""

import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.db import fetchone_dict
from models.ranking.feature_engineer import build_training_df
from models.ranking.custom_deepfm import DeepFM

os.makedirs("saved_model", exist_ok=True)
MODEL_PATH = "saved_model/deepfm_ranker.pt"


# ------------------------------------------------------------------ #
#                    vocab size (user / movie / genre)               #
# ------------------------------------------------------------------ #
def _vocab_sizes():
    mu = fetchone_dict("SELECT MAX(id) AS m FROM users")["m"] or 0
    mm = fetchone_dict("SELECT MAX(id) AS m FROM movies")["m"] or 0
    mg = fetchone_dict("SELECT MAX(id) AS m FROM genres")["m"] or 0
    return [mu + 2, mm + 2, mg + 2]          # +2 for padding / OOV


# ------------------------------------------------------------------ #
#                       helper → TensorDataset                       #
# ------------------------------------------------------------------ #
def _to_tensor(df, sparse_cols, dense_cols):
    Xs = torch.tensor(df[sparse_cols].values, dtype=torch.long)
    Xd = torch.tensor(df[dense_cols ].values, dtype=torch.float32)
    y  = torch.tensor(df["label"].values,   dtype=torch.float32)
    return TensorDataset(Xs, Xd, y)


# ------------------------------------------------------------------ #
#                               main                                 #
# ------------------------------------------------------------------ #
def main(epochs=3, batch_size=2048, neg_ratio=1):

    df = build_training_df(neg_ratio)
    if df.empty:
        print("[train_ranking] ❌ 无训练样本")
        return

    # -------- 稀疏 & 稠密特征列 --------
    sparse_cols = ["user_id", "movie_id", "genre_id"]
    dense_cols  = ["recall_score", "vote_average", "popularity", "age"]

    # 训练阶段 recall_score 不存在 → 填 0.0
    df["recall_score"] = 0.0

    # -------- train / val split --------
    tr_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(_to_tensor(tr_df, sparse_cols, dense_cols),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_to_tensor(val_df, sparse_cols, dense_cols),
                              batch_size=batch_size, shuffle=False)

    # -------- 构建模型 --------
    field_dims = _vocab_sizes()
    model = DeepFM(field_dims,
                   num_dense=len(dense_cols),
                   embed_dim=16,
                   mlp_dims=(128, 64),
                   dropout=0.2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn  = nn.BCEWithLogitsLoss()

    print(f"[train_ranking] 样本={len(df)}  正负≈1:{neg_ratio}")
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tot, n = 0., 0
        for Xs, Xd, y in tqdm(train_loader, desc=f"Ep {ep}/{epochs}", ncols=80):
            Xs, Xd, y = Xs.to(device), Xd.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xs, Xd), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n   += len(y)
        print(f"  train_loss={tot/n:.4f}")

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            tot_v, nv = 0., 0
            for Xs, Xd, y in val_loader:
                Xs, Xd, y = Xs.to(device), Xd.to(device), y.to(device)
                tot_v += loss_fn(model(Xs, Xd), y).item() * len(y)
                nv    += len(y)
        print(f"  val_loss  ={tot_v/nv:.4f}")

    # -------- Save --------
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    print("✅ DeepFM 已保存 →", MODEL_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int, default=3)
    ap.add_argument("--batch",      type=int, default=2048)
    ap.add_argument("--neg_ratio",  type=int, default=1)
    args = ap.parse_args()
    main(args.epochs, args.batch, args.neg_ratio)

# models/ranking/infer_ranking.py
"""
用自定义 DeepFM 对召回候选精排
"""

import numpy as np
import torch

from models.db import fetchone_dict
from models.ranking.feature_engineer import build_infer_df
from models.ranking.custom_deepfm import DeepFM

MODEL_PATH = "saved_model/deepfm_ranker.pt"


def _vocab_sizes():
    mu = fetchone_dict("SELECT MAX(id) AS m FROM users")["m"] or 0
    mm = fetchone_dict("SELECT MAX(id) AS m FROM movies")["m"] or 0
    mg = fetchone_dict("SELECT MAX(id) AS m FROM genres")["m"] or 0
    return [mu + 2, mm + 2, mg + 2]


def _load_model(field_dims, num_dense):
    m = DeepFM(field_dims, num_dense)
    try:
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        return m
    except FileNotFoundError:
        return None


def rank_candidates(user_id, movie_ids, recall_scores, top_n=10):
    if not movie_ids:
        return []

    df = build_infer_df(user_id, movie_ids, recall_scores)

    sparse_cols = ["user_id", "movie_id", "genre_id"]
    dense_cols  = ["recall_score", "vote_average", "popularity", "age"]

    field_dims  = _vocab_sizes()
    model = _load_model(field_dims, num_dense=len(dense_cols))

    if model is None:
        # 模型未训练：按召回分降序
        idx = np.argsort(-np.array(recall_scores))[:top_n]
        return list(np.array(movie_ids)[idx])

    xs = torch.tensor(df[sparse_cols].values, dtype=torch.long)
    xd = torch.tensor(df[dense_cols ].values, dtype=torch.float32)

    with torch.no_grad():
        score = torch.sigmoid(model(xs, xd)).numpy()

    df["score"] = score
    return (df.sort_values("score", ascending=False)
              .head(top_n)["movie_id"].tolist())


# -- quick CLI test -----------------
if __name__ == "__main__":
    import argparse
    from models.recall.two_tower import load_model, recommend_warm_start
    from models.db import get_movie_titles

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", type=int)
    args = ap.parse_args()

    tower = load_model()
    cand_ids, cand_scores = recommend_warm_start(tower, args.user_id, top_n=300)

    ranked = rank_candidates(args.user_id, cand_ids, cand_scores, top_n=10)
    titles = get_movie_titles(ranked)
    for i, mid in enumerate(ranked, 1):
        print(f"{i:02}. {titles.get(mid,'Unknown')} (ID={mid})")

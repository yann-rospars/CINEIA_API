# . models/ranking/feature_engineer.py

"""
把数据库字段整理成 DeepCTR-Torch 需要的 DataFrame
仅使用 *现有* 表：users · movies · movie_genre · view_history
"""

from collections import defaultdict
from typing import List, Tuple

import pandas as pd

from models.db import fetchall_dict, fetchone_dict


# ------------------------------------------------------------------- #
#                        基础静态特征拉取                             #
# ------------------------------------------------------------------- #
def _get_movie_features() -> pd.DataFrame:
    """
    movie_id | genre_id | vote_average | popularity
    若一部电影有多 genre，只取第一条；可后续改成多值特征
    """
    # genre
    rows = fetchall_dict("""
        SELECT movie_id, genre_id
        FROM movie_genre
        ORDER BY movie_id, genre_id
    """)
    first_genre = {}
    for r in rows:
        first_genre.setdefault(r["movie_id"], r["genre_id"])

    # movies main table
    rows = fetchall_dict("""
        SELECT id AS movie_id,
               vote_average,
               popularity
        FROM movies
    """)
    for r in rows:
        r["genre_id"] = first_genre.get(r["movie_id"], 0)
    return pd.DataFrame(rows)


def _get_user_features() -> pd.DataFrame:
    """
    user_id | age
    """
    rows = fetchall_dict("SELECT id AS user_id, COALESCE(age, 0) AS age FROM users")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- #
#                       训练数据构造 (label)                           #
# ------------------------------------------------------------------- #
def build_training_df(neg_ratio: int = 1) -> pd.DataFrame:
    """
    label = 1 -> view_history 里的正样本
    label = 0 -> 随机负采样 (未观看)
    """
    pos_rows = fetchall_dict("SELECT user_id, movie_id FROM view_history")
    if not pos_rows:                       # 数据不足
        return pd.DataFrame()

    pos_df = pd.DataFrame(pos_rows)
    pos_df["label"] = 1

    # Negative samples ------------------------------------------------
    all_movies = {r["movie_id"] for r in fetchall_dict("SELECT id AS movie_id FROM movies")}
    user_pos = defaultdict(set)
    for r in pos_rows:
        user_pos[r["user_id"]].add(r["movie_id"])

    import random
    neg_records = []
    for u, watched in user_pos.items():
        cand = list(all_movies - watched)
        k = min(len(cand), neg_ratio * len(watched))
        for m in random.sample(cand, k):
            neg_records.append({"user_id": u, "movie_id": m, "label": 0})
    neg_df = pd.DataFrame(neg_records)

    data_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 加入静态特征 ------------------------------------------------------
    movies_df = _get_movie_features()
    users_df = _get_user_features()
    df = data_df.merge(movies_df, on="movie_id", how="left") \
                .merge(users_df, on="user_id", how="left")

    # 若有缺失，用 0 填充
    df.fillna(0, inplace=True)
    return df


# ------------------------------------------------------------------- #
#                         推断特征构造                                #
# ------------------------------------------------------------------- #
def build_infer_df(user_id: int,
                   movie_ids: List[int],
                   recall_scores: List[float]
                   ) -> pd.DataFrame:
    """
    组装推断时特征。包含 recall_score 列！
    """
    movies_df = _get_movie_features()
    users_df  = _get_user_features()

    # 若用户不存在，补 age=0
    if user_id not in users_df["user_id"].values:
        users_df = pd.concat([users_df,
                              pd.DataFrame([{"user_id": user_id, "age": 0}])])

    infer_df = pd.DataFrame({
        "user_id":      user_id,
        "movie_id":     movie_ids,
        "recall_score": recall_scores,
    })

    infer_df = (infer_df
                .merge(movies_df, on="movie_id", how="left")
                .merge(users_df,  on="user_id", how="left"))

    infer_df.fillna(0, inplace=True)
    return infer_df

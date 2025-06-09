# models/recall/two_tower.py
"""
Two-Tower 召回模型推断逻辑

把原先的 warm_start.py + infer.py 合并成一个文件，供 service 层调用：
    from models.recall.two_tower import load_model, recommend_warm_start
"""

import os
import time
from typing import List, Tuple

import numpy as np
import torch

from models.db import (
    get_max_user_id,
    get_max_movie_id,
    get_all_movie_ids_with_language,
    get_user_view_languages,
)
from models.pytorch_model import TwoTowerMLPModel


# --------------------------------------------------------------------------- #
#                           模型加载 / 缓存                                    #
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}  # {"path": model}


from pathlib import Path

def load_model(model_path: str = None, embedding_dim: int = 32) -> TwoTowerMLPModel:
    """
    读取并缓存 Two-Tower 模型；多次调用不会重复 load。
    """
    # 如果没有指定路径，默认用绝对路径
    if model_path is None:
        CURRENT_DIR = Path(__file__).resolve().parent  # DNN_TorchFM_TTower/models/recall
        model_path = CURRENT_DIR.parent.parent / 'saved_model' / 'dnn_recommender.pt'

    model_path = Path(model_path)

    if str(model_path) in _MODEL_CACHE:
        return _MODEL_CACHE[str(model_path)]

    if not model_path.exists():
        raise FileNotFoundError(
            f"[two_tower] 模型文件 {model_path} 不存在，请先训练再推断。"
        )

    max_u, max_m = get_max_user_id(), get_max_movie_id()
    model = TwoTowerMLPModel(max_u, max_m,
                             embedding_dim=embedding_dim,
                             hidden_dim=64)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()

    _MODEL_CACHE[str(model_path)] = model
    return model



# --------------------------------------------------------------------------- #
#                           推 断 主 逻 辑                                     #
# --------------------------------------------------------------------------- #
def recommend_warm_start(model: TwoTowerMLPModel,
                         user_id: int,
                         top_n: int = 10
                         ) -> Tuple[List[int], List[float]]:
    """
    针对“有历史行为”的用户，使用 Two-Tower 模型做召回预测。

    Returns
    -------
    movie_ids : List[int]
        Top-N 电影 id
    scores    : List[float]
        对应 sigmoid 分数（概率），可作为粗排得分
    """
    tic = time.time()

    # 1) 根据用户偏好语言做 candidate 下采样
    preferred_langs = get_user_view_languages(user_id)
    all_movies = get_all_movie_ids_with_language()

    if preferred_langs:
        candidate_movies = [m for m, lang in all_movies
                            if lang in preferred_langs]
    else:
        candidate_movies = [m for m, _ in all_movies]

    if not candidate_movies:
        return [], []

    # 2) 模型推断
    movie_tensor = torch.tensor(candidate_movies, dtype=torch.long)
    user_tensor = torch.full((len(candidate_movies),),
                             user_id, dtype=torch.long)

    with torch.no_grad():
        logits = model(user_tensor, movie_tensor)
        scores = torch.sigmoid(logits).numpy().flatten()

    # 3) 取 Top-N
    top_idx = np.argsort(-scores)[:top_n]
    top_movie_ids = np.array(candidate_movies)[top_idx]
    top_scores = scores[top_idx]

    print(f"[two_tower] Inference finished in {time.time() - tic:.2f}s")
    return top_movie_ids.tolist(), top_scores.tolist()


# --------------------------------------------------------------------------- #
#                              命令行测试                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from models.db import get_movie_titles

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", type=int)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    mdl = load_model()
    mids, scs = recommend_warm_start(mdl, args.user_id, args.top_n)
    title_map = get_movie_titles(mids)

    print(f"\nTop {args.top_n} for user {args.user_id}:")
    for mid, s in zip(mids, scs):
        print(f"  {mid:<6}  {title_map.get(mid, 'Unknown'):<40}  {s:.4f}")

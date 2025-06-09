# service/recommender.py

"""
统一推荐入口：
    • 新用户       → 冷启动 (PopRec + 随机多样化)
    • 老用户       → Two-Tower 召回  → DeepFM 精排
对上层调用者隐藏实现细节，只暴露一个函数 `recommend_movies_for_user`
"""

from __future__ import annotations

import functools
import time
from typing import List, Tuple

from models.db import get_user_view_count
from models.recall import cold_start
from models.recall.two_tower import load_model as _load_tower, recommend_warm_start
from models.ranking.infer_ranking import rank_candidates

# --------------------------------------------------------------------------- #
#                          全局缓存 —— 只加载一次                              #
# --------------------------------------------------------------------------- #
_TOWER_MODEL = functools.lru_cache(maxsize=1)(_load_tower)()   # type: ignore


# --------------------------------------------------------------------------- #
#                          核 心 接 口                                        #
# --------------------------------------------------------------------------- #
def recommend_movies_for_user(user_id: int,
                              n_recall: int = 300,
                              n_final: int = 20
                              ) -> List[int]:
    """
    返回最终推荐的 movie_id 列表 (长度 n_final，不含重复)。
    若精排模型未训练，则直接按召回分排序。
    """
    t0 = time.time()
    view_cnt = get_user_view_count(user_id)

    # -------- 冷启动 --------
    if view_cnt == 0:
        print(f"[recommender] user {user_id} cold-start")
        movie_ids = cold_start.recommend_cold_start(top_n=n_final)
        print(f"[recommender] cold-start done ({len(movie_ids)} items, {time.time()-t0:.2f}s)")
        return movie_ids

    # -------- 热启动：召回 --------
    recall_ids, recall_scores = recommend_warm_start(
        _TOWER_MODEL, user_id, top_n=n_recall
    )
    if not recall_ids:
        # 回退：冷启动
        print(f"[recommender] warm-start empty, fallback to cold")
        return cold_start.recommend_cold_start(top_n=n_final)

    # -------- 精排 --------
    ranked = rank_candidates(user_id, recall_ids, recall_scores, top_n=n_final)
    print(f"[recommender] warm-start+rank done ({len(ranked)} items, {time.time()-t0:.2f}s)")
    return ranked


# --------------------------------------------------------------------------- #
#                           CLI 测 试                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from models.db import get_movie_titles

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", type=int)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    mids = recommend_movies_for_user(args.user_id, n_final=args.top)
    titles = get_movie_titles(mids)

    print("\n推荐结果：")
    for i, mid in enumerate(mids, 1):
        print(f"{i:02}. {titles.get(mid, 'Unknown')}  (ID={mid})")

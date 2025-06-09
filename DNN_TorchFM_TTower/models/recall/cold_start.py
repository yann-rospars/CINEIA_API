# models/recall/cold_start.py
import random
from models.db import fetchall_dict

def recommend_cold_start(top_n=10):
    """
    针对无历史用户(冷启动)：先选一批高评分热门电影，再随机抽取 top_n。
    示例：我们假设在 movies 表中有 vote_average, vote_count 字段。
    """
    query = """
        SELECT id
        FROM movies
        WHERE vote_count > 0
        ORDER BY vote_average DESC, vote_count DESC
        LIMIT 50
    """
    rows = fetchall_dict(query)
    if not rows:
        return []
    candidate_ids = [r["id"] for r in rows]
    
    # 从这 50 部评分最高的影片里随机抽 top_n
    if len(candidate_ids) <= top_n:
        return candidate_ids
    return random.sample(candidate_ids, top_n)

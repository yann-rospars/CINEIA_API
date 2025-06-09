#!/usr/bin/env python3
"""
/scripts/incremental.py
Interactive CLI demo:
1. 如果用户不存在则自动插入 users 表
2. 冷启动推荐 Top-K 让你选择想看的片
3. 把选择写入 view_history
4. 触发一次增量训练（可选）
5. 走 Warm Start + 精排，再推荐下一批
6. 支持循环，直到用户输入 q 退出
"""

import readline  # 为了 ↑↓ 历史
from models.db import (
    fetchone_dict, execute_sql,
    get_movie_titles, get_user_view_count
)
from models.recall.cold_start import recommend_cold_start
from models.recall.two_tower import load_model as load_recall_model
from models.recall.two_tower import recommend_warm_start
from models.ranking.infer_ranking import rank_candidates
from models.recall.train_incremental import incremental_train

RECALL_MODEL = load_recall_model()
USER_ID = int(input("请输入用户 ID（新用户填一个从未用过的数字）：").strip())

def ensure_user(uid):
    if not fetchone_dict("SELECT 1 FROM users WHERE id=%s", (uid,)):
        execute_sql(
            "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s)",
            (uid, f"cli_{uid}@demo.com", "hash_placeholder")
        )
        print(f"✅ 已创建新用户 {uid}")

def insert_views(uid, movie_ids):
    for mid in movie_ids:
        execute_sql("INSERT INTO view_history (user_id, movie_id) VALUES (%s, %s)", (uid, mid))

def choose_movies(candidates, title_map):
    print("\n请在下面输入想看的电影编号，用空格分隔（回车结束，q 返回退出）：")
    for i, mid in enumerate(candidates, 1):
        print(f"[{i}] {title_map.get(mid, 'Unknown')}")
    while True:
        raw = input("你的选择: ").strip()
        if raw.lower() in {"q", "quit"}:
            return None
        try:
            idxs = [int(s) for s in raw.split()]
            chosen = [candidates[i-1] for i in idxs if 1 <= i <= len(candidates)]
            return chosen
        except ValueError:
            print("❌ 输入格式错误，请重新输入")

def main_loop():
    ensure_user(USER_ID)

    while True:
        watch_cnt = get_user_view_count(USER_ID)
        print(f"\n=== 当前观影次数: {watch_cnt} ===")

        if watch_cnt == 0:
            # ---- Cold Start ----
            rec_ids = recommend_cold_start(top_n=10)
            phase = "冷启动"
        else:
            # ---- Recall + Ranking ----
            rec_ids, recall_scores = recommend_warm_start(RECALL_MODEL, USER_ID, top_n=300)
            rec_ids = rank_candidates(USER_ID, rec_ids, recall_scores, top_n=10)
            phase = "热启动 (召回+精排)"

        title_map = get_movie_titles(rec_ids)
        print(f"\n【{phase}】为你推荐:")
        for i, mid in enumerate(rec_ids, 1):
            print(f"  {i}. {title_map.get(mid, 'Unknown')} (MovieID={mid})")

        chosen = choose_movies(rec_ids, title_map)
        if chosen is None:
            print("👋 再见！")
            break

        insert_views(USER_ID, chosen)
        print(f"✅ 已记录 {len(chosen)} 部影片观看历史。")

        # 每次循环小规模增量训练一下（真实线上可异步 / 定时）
        incremental_train(neg_ratio=1, epochs=1)

if __name__ == "__main__":
    main_loop()

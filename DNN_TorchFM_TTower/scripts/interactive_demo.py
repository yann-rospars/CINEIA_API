#!/usr/bin/env python3
"""
scripts/interactive_demo.py
CLI 交互式推荐演示 interactive demo

流程：
  1. 输入一个用户 ID（若不存在自动插入）
  2. 调用 service.recommender 推荐 TOP_N 部电影
  3. 用户手动选择若干部想看的 → 写入 view_history
  4. 每 LOOP_FOR_RETRAIN 轮：
        • 召回侧增量训练 1 epoch
        • 精排侧增量训练 1 epoch
  5. 回到步骤 2，直到输入 q / quit 退出
"""

from typing import List

# ------- readline 在 Linux / macOS 默认有；Windows 没有也不影响 -------
try:
    import readline  # noqa: F401
except ImportError:
    pass

from models.db import (
    fetchone_dict, execute_sql,
    get_movie_titles, get_user_view_count,
)

from models.recall.train_incremental import incremental_train as recall_inc_train
from models.ranking.train_ranking import main as ranking_train_main
from service.recommender import recommend_movies_for_user

TOP_N = 10
LOOP_FOR_RETRAIN = 3      # 每 3 轮做一次增量重训（可调为 0 关闭）


# ------------------------------------------------------------------ #
#                 DB 帮助函数：建用户 / 插观看记录                    #
# ------------------------------------------------------------------ #
def ensure_user(user_id: int):
    if not fetchone_dict("SELECT 1 FROM users WHERE id=%s", (user_id,)):
        execute_sql(
            "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s)",
            (user_id, f"cli_{user_id}@demo.com", b"hash_placeholder"),
        )
        print(f"✅ new user created {user_id}")


def insert_views(user_id: int, movie_ids: List[int]):
    for mid in movie_ids:
        execute_sql(
            "INSERT INTO view_history (user_id, movie_id) VALUES (%s, %s)",
            (user_id, mid),
        )


# ------------------------------------------------------------------ #
#                             交互函数                               #
# ------------------------------------------------------------------ #
def choose_movies(candidates: List[int], titles: dict[int, str]) -> List[int] | None:
    print("\nPlease enter the serial number of the movie you want to watch (separated by Spaces), or exit with q:")
    for i, m in enumerate(candidates, 1):
        print(f"[{i:02}] ID={m} | {titles.get(m, 'Unknown')} ")

    while True:
        raw = input("your choices: ").strip().lower()
        if raw in {"q", "quit"}:
            return None
        try:
            idxs = [int(s) for s in raw.split()]
            chosen = [candidates[i - 1] for i in idxs if 1 <= i <= len(candidates)]
            return chosen
        except ValueError:
            print("The input format is incorrect. Please re-enter")


# ------------------------------------------------------------------ #
#                         增量训练封装                                #
# ------------------------------------------------------------------ #
def incremental_retrain():
    print("\n Incremental training begins (recall 1 epoch + rerank 1 epoch)...")
    recall_inc_train(neg_ratio=1, epochs=1)
    ranking_train_main(epochs=1, batch_size=4096, neg_ratio=1)
    print("incremental training fini\n")


# ------------------------------------------------------------------ #
#                           主循环                                   #
# ------------------------------------------------------------------ #
def interactive_loop(user_id: int):
    ensure_user(user_id)
    loop_cnt = 0

    while True:
        viewed = get_user_view_count(user_id)
        print(f"\n=== user {user_id} has watched {viewed} films ===")

        mids, scores, strategy = recommend_movies_for_user(user_id, n_final=TOP_N)
        if not mids:
            print("  unable to fetch datas, please check the database connection")
            break

        title_map = get_movie_titles(mids)
        chosen = choose_movies(mids, title_map)
        if chosen is None:
            print("\n👋 Bye~")
            break

        insert_views(user_id, chosen)
        print(f" 已记录观看 {len(chosen)} 部影片。")
        loop_cnt += 1

        # ---- 条件触发增量训练 ----
        if LOOP_FOR_RETRAIN and loop_cnt % LOOP_FOR_RETRAIN == 0:
            incremental_retrain()



# ------------------------------------------------------------------ #
#                                 main                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    try:
        uid = int(input("Please enter the user ID (fill in any integer for a new one) : ").strip())
    except ValueError:
        print(" x The user ID must be an integer")
        exit(1)

    interactive_loop(uid)

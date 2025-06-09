# DNN_TorchFM_TTower\models\db.py
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import Counter
from models.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def fetchall_dict(query, params=None):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()

def fetchone_dict(query, params=None):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchone()

def get_movie_titles(movie_ids):
    query = "SELECT id, title FROM movies WHERE id = ANY(%s)"
    rows = fetchall_dict(query, (movie_ids,))
    return {row["id"]: row["title"] for row in rows}

def get_max_user_id():
    row = fetchone_dict("SELECT MAX(id) as m FROM users")
    return row["m"] or 0

def get_max_movie_id():
    row = fetchone_dict("SELECT MAX(id) as m FROM movies")
    return row["m"] or 0

def get_all_movie_ids_with_language():
    """
    返回 (电影id, original_language) 元组列表，仅返回有 original_language 的记录。
    """
    rows = fetchall_dict("SELECT id, original_language FROM movies")
    return [(r["id"], r["original_language"]) for r in rows if r["original_language"]]

def get_user_view_languages(user_id):
    """
    根据 view_history 表，统计用户看过的电影的语言分布，并取最常见的前2种语言。
    """
    query = """
        SELECT m.original_language
        FROM view_history v
        JOIN movies m ON v.movie_id = m.id
        WHERE v.user_id = %s
    """
    rows = fetchall_dict(query, (user_id,))
    lang_counter = Counter(r["original_language"] for r in rows if r["original_language"])
    if not lang_counter:
        return set()
    # 假设我们只关心最常见的2种语言
    top_languages = {lang for lang, _ in lang_counter.most_common(2)}
    return top_languages

def get_user_view_count(user_id):
    """
    返回用户在 view_history 表中的观影数量。若为 0，则表示新用户（无历史记录）。
    """
    query = "SELECT COUNT(*) as cnt FROM view_history WHERE user_id = %s"
    row = fetchone_dict(query, (user_id,))
    return row["cnt"] if row else 0

def get_top_rated_movies(limit=10):
    """
    根据 movies 表里的 vote_average 字段，获取评分最高的电影 ID 列表。
    如果想考虑投票数量，也可以按 (vote_average DESC, vote_count DESC) 来排序。
    """
    query = """
        SELECT id
        FROM movies
        WHERE vote_count > 0
        ORDER BY vote_average DESC, vote_count DESC
        LIMIT %s
    """
    rows = fetchall_dict(query, (limit,))
    return [r["id"] for r in rows]

def execute_sql(query, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()

def get_movie_metadata(movie_ids: list[int]) -> dict[int, dict]:
    if not movie_ids:
        return {}

    query = f"""
        SELECT id, title, poster_path
        FROM movies
        WHERE id = ANY(%s)
    """
    results = fetchall_dict(query, (movie_ids,))
    return {
        row["id"]: {
            "title": row["title"],
            "poster": row['poster_path']
        }
        for row in results
    }

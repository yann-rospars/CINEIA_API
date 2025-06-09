from flask import Flask, request, jsonify
import os
from minio import Minio

# ────────────────────────────────────────────────────────────── #
# Téléchargement automatique des modèles depuis MinIO
# ────────────────────────────────────────────────────────────── #
def download_from_minio_if_needed(bucket_name, object_name, target_path):
    if not os.path.exists(target_path):
        print(f"Téléchargement de {object_name} depuis MinIO...")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        client = Minio(
            "45.149.207.13:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
        client.fget_object(bucket_name, object_name, target_path)
    else:
        print(f"Modèle déjà présent : {target_path}")

# Télécharger les deux modèles si absents
download_from_minio_if_needed("models", "dnn_recommender.pt", "saved_model/dnn_recommender.pt")
download_from_minio_if_needed("models", "deepfm_ranker.pt", "saved_model/deepfm_ranker.pt")

# ────────────────────────────────────────────────────────────── #
# Imports (à faire après le téléchargement des modèles)
# ────────────────────────────────────────────────────────────── #
from models.db import get_user_view_count, get_movie_metadata
from models.recall.cold_start import recommend_cold_start
from service.recommender import recommend_movies_for_user

# ────────────────────────────────────────────────────────────── #
# API Flask
# ────────────────────────────────────────────────────────────── #
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    user_id_param = request.args.get("user_id")

    try:
        user_id = int(user_id_param) if user_id_param else None
    except ValueError:
        return jsonify({"error": "user_id invalide"}), 400

    try:
        if user_id is None or get_user_view_count(user_id) == 0:
            movie_ids = recommend_cold_start(top_n=10)
            metadata = get_movie_metadata(movie_ids)
            result = [
                {
                    "id": mid,
                    "title": metadata.get(mid, {}).get("title", "Unknown"),
                    "poster": f"https://image.tmdb.org/t/p/w500{metadata.get(mid, {}).get('poster')}"
                    if metadata.get(mid, {}).get("poster") else None,
                    "strategy": "cold"
                }
                for mid in movie_ids
            ]
        else:
            movie_ids, scores, strategy = recommend_movies_for_user(user_id, n_final=10)
            metadata = get_movie_metadata(movie_ids)
            result = [
                {
                    "id": mid,
                    "title": metadata.get(mid, {}).get("title", "Unknown"),
                    "poster": f"https://image.tmdb.org/t/p/w500{metadata.get(mid, {}).get('poster')}"
                    if metadata.get(mid, {}).get("poster") else None,
                    "score": round(score, 4) if score else None,
                    "strategy": strategy
                }
                for mid, score in zip(movie_ids, scores)
            ]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

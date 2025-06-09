# CINEIA – AI Back-End (Two-Tower Recall + DeepFM Re-Rank)

This branch contains the complete Python code required to **train**, **serve** and **try** the recommendation engine that will later be queried by the front-end.

###  Quick start

```bash
# 0  move into the project
cd DNN_TorchFM_TTower

# 1  install dependencies (CPU PyTorch by default)
pip install --upgrade -r requirements.txt

# 2  (optional) edit models/config.py to point to your PostgreSQL

# 3  train / retrain the Two-Tower recall model
python -m DNN_TorchFM_TTower.models.recall.train_two_tower  --epochs 3 --batch 128

# 4  train / retrain the DeepFM re-rank model
python -m DNN_TorchFM_TTower.models.ranking.train_ranking   --epochs 3

# 5  run the interactive CLI demo (cold-start → warm-start → incremental retrain)
python -m DNN_TorchFM_TTower.scripts.interactive_demo

```

### Project layout (high level)
```bash
models/
│
├─ config.py             ← DB credentials
├─ db.py                 ← thin PostgreSQL helper
├─ pytorch_model.py      ← Two-Tower network
│
├─ recall/               ← coarse recall layer
│   ├─ cold_start.py
│   ├─ two_tower.py      ← inference helper
│   ├─ train_two_tower.py
│   └─ train_incremental.py
│
└─ ranking/              ← fine re-rank layer
    ├─ custom_deepfm.py  ← pure-PyTorch DeepFM
    ├─ feature_engineer.py
    ├─ train_ranking.py
    └─ infer_ranking.py
service/
│   ├─ recommender.py    ← single python entry, returns Top-N ids
│   └─ api.py (optional) ← FastAPI REST wrapper
scripts/
    └─ interactive_demo.py
saved_model/             ← trained weights (auto-created)
requirements.txt
```

### AI pipeline
```bash
new user
   │ cold_start(pop-REC+diversity)
   ▼
movie ids ───────────────┐
                         │
old user                 ▼
(view history) → Two-Tower recall (300 ids, score)
                         │
                         ▼
                 DeepFM re-rank (uses recall_score + 3 dense feats)
                         │
                         ▼
                    Top-N  personalised list
```

### Database quick reference
-- inspect public schema
```sql
SELECT table_name   AS "Table",
       column_name  AS "Column",
       data_type    AS "Type",
       is_nullable  AS "NULL?",
       column_default AS "Default"
FROM   information_schema.columns
WHERE  table_schema = 'public'
ORDER  BY table_name, ordinal_position;
```